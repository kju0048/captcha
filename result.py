from fastapi import FastAPI, Response
import os
import numpy as np
from PIL import Image
import torch as th
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import requests
import json


# 필요한 모듈 임포트
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

app = FastAPI()

def sample_images(args):
    """이미지 샘플을 생성하고 .npz 파일로 저장합니다."""
    # 분산 설정을 단일 프로세스로 간소화
    dist_util.dev = lambda: th.device('cpu')
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_path = args.model_path
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    logger.log(f"loading checkpoint: {model_path}")
    logger.log(f"timesteps: {args.timestep_respacing}")
    model.to(th.device('cpu'))
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=th.device('cpu')
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.append(sample.cpu().numpy())
        if args.class_cond:
            all_labels.append(classes.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join("outputs", f"samples_{shape_str}.npz")
    os.makedirs("outputs", exist_ok=True)
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)
    return out_path

def save_images_from_npz(npz_path, output_dir, target_size=(128, 128)):
    """npz 파일에서 이미지를 추출하고 크기를 조정하여 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    images = data["arr_0"]
    image_filenames = []
    for i, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        img = img.resize(target_size, Image.LANCZOS)  # Resize to target_size
        img_filename = f"output_image_{i}.jpg"
        img.save(os.path.join(output_dir, img_filename))
        image_filenames.append(img_filename)
    print(f"Saved and resized images to: {output_dir}")
    return image_filenames

def load_model(od_model_path):
    """사전 학습된 모델을 로드하고 state_dict 키를 재매핑합니다."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 6  # 0: 배경, 1~5: 우리의 클래스
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the saved state_dict
    state_dict = th.load(od_model_path, map_location='cpu')
    new_state_dict = {}

    # Remap keys to match the current model structure
    for key, value in state_dict.items():
        # Remap "inner_blocks", "layer_blocks", and "conv" keys
        new_key = key.replace(".0.0", ".0").replace(".1.0", ".1").replace(".2.0", ".2").replace(".3.0", ".3")
        new_key = new_key.replace("rpn.head.conv.0.", "rpn.head.conv.")  # Fix RPN conv keys
        new_state_dict[new_key] = value

    # Load the remapped state_dict
    model.load_state_dict(new_state_dict)
    model.to('cpu')
    model.eval()
    return model

def infer_images(model, image_dir, input_size=(128, 128)):
    """리사이즈된 이미지에서 얼굴 랜드마크를 추론하고 바운딩 박스의 중심을 저장합니다."""
    results = []
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        with th.no_grad():
            output = model(img_tensor)[0]  # Faster R-CNN returns a list; get the first image output

        # Initialize a result dictionary for this image
        result = {}  # 'image_id'를 포함하지 않음

        # Ensure there are detections
        if "boxes" in output and len(output["boxes"]) > 0:
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                # Skip detections with low confidence
                if score.item() < 0.5:
                    continue

                # Calculate the center of the bounding box
                center_x = (box[0].item() + box[2].item()) / 2
                center_y = (box[1].item() + box[3].item()) / 2

                # Map labels to landmark names and save the center values
                if label.item() == 1:  # lefteye
                    result["lefteye_x"] = round(center_x)
                    result["lefteye_y"] = round(center_y)
                elif label.item() == 2:  # righteye
                    result["righteye_x"] = round(center_x)
                    result["righteye_y"] = round(center_y)
                elif label.item() == 3:  # nose
                    result["nose_x"] = round(center_x)
                    result["nose_y"] = round(center_y)
                elif label.item() == 4:  # leftmouth
                    result["leftmouth_x"] = round(center_x)
                    result["leftmouth_y"] = round(center_y)
                elif label.item() == 5:  # rightmouth
                    result["rightmouth_x"] = round(center_x)
                    result["rightmouth_y"] = round(center_y)

        result["z_all_count"] = 1
        result["z_correct_count"] = 0
        result["z_incorrect_count"] = 0
        results.append((img_name, result))  # 이미지 이름과 결과를 함께 저장
    return results

def get_database_data():
    """Firebase Realtime Database에서 전체 데이터를 가져옵니다."""
    database_url = "https://aifront-a7a19-default-rtdb.firebaseio.com/.json"

    response = requests.get(database_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to get data from Firebase: {response.text}")
        return {}

def update_database_data(data):
    """Firebase Realtime Database에 전체 데이터를 업데이트합니다."""
    database_url = "https://aifront-a7a19-default-rtdb.firebaseio.com/.json"

    response = requests.put(database_url, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Failed to update data in Firebase: {response.text}")
    else:
        print("Database updated successfully.")

def get_next_image_index(data):
    """'counter'의 'count' 값을 가져오고 증가시킵니다."""
    if "counter" not in data or not isinstance(data["counter"], dict):
        data["counter"] = {"count": 0}
    count = data["counter"].get("count", 0)
    data["counter"]["count"] = count + 1  # 'count' 값을 1 증가
    return count  # 증가 전의 값을 반환하여 인덱스로 사용

def upload_image_to_firebase(local_img_path, storage_img_name):
    """이미지를 Firebase Storage에 업로드합니다."""
    bucket_name = "aifront-a7a19.appspot.com"
    folder_name = "image2"
    upload_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o?name={folder_name}%2F{storage_img_name}"

    headers = {
        "Content-Type": "application/octet-stream",
    }
    params = {
        "uploadType": "media",
    }

    with open(local_img_path, 'rb') as img_file:
        img_data = img_file.read()

    response = requests.post(upload_url, headers=headers, params=params, data=img_data)
    if response.status_code != 200:
        print(f"Failed to upload {storage_img_name}: {response.text}")
    else:
        print(f"Uploaded {storage_img_name} successfully.")

def save_data_to_firebase(data, image_id, result, is_def=False):
    """데이터를 Firebase Realtime Database에 저장합니다."""
    key = "def" if is_def else "gen"
    if key not in data or not isinstance(data[key], dict):
        data[key] = {}

    # 이미지 ID를 키로 사용하여 데이터를 저장
    image_id_key = image_id.replace('.', '_')  # '.'을 '_'로 대체하여 키로 사용
    data[key][image_id_key] = result  # 'image_id'를 포함하지 않는 result 저장
    print(f"Saved data for {image_id} to '{key}' with key '{image_id_key}'.")

@app.get("/run")
def run_pipeline():
    # Step 1: 기본 인자 설정
    from argparse import Namespace
    from guided_diffusion.script_util import model_and_diffusion_defaults

    # 기본값을 가져옵니다.
    defaults = model_and_diffusion_defaults()
    # 필요한 추가 인자들을 설정합니다.
    defaults.update({
        "clip_denoised": True,
        "num_samples": 1,
        "batch_size": 1,
        "use_ddim": False,
        "model_path": "./model/DDPM_IP_celeba64.pt",
        "timestep_respacing": "100",
        "class_cond": False,
        "use_fp16": False,  # CPU에서 실행하므로 False로 설정
        "image_size": 64,
        "num_channels": 192,
        "num_res_blocks": 3,
        "attention_resolutions": "32,16,8",
        "num_heads": 4,
        "num_head_channels": 64,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "learn_sigma": True,
        "dropout": 0.1,
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "use_scale_shift_norm": True,
        "rescale_learned_sigmas": True,
    })

    # Namespace 객체로 변환합니다.
    args = Namespace(**defaults)

    # Step 4: 사전 학습된 얼굴 랜드마크 모델 로드
    od_model_path = "./model/detection.pth"
    model = load_model(od_model_path)

    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # output_dir을 정리합니다.
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        # Step 2: 이미지 샘플링 및 .npz로 저장
        npz_path = sample_images(args)

        # Step 3: .npz를 리사이즈된 .jpg로 변환
        image_filenames = save_images_from_npz(npz_path, output_dir, target_size=(128, 128))

        # Step 5: 추론 수행
        results = infer_images(model, output_dir, input_size=(128, 128))

        # 결과가 있는지 확인
        if results:
            img_name, result = results[0]  # 생성된 이미지는 하나이므로 첫 번째 결과를 사용
            # 필요한 랜드마크 키 목록
            required_landmarks = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y",
                                  "nose_x", "nose_y", "leftmouth_x", "leftmouth_y",
                                  "rightmouth_x", "rightmouth_y"]
            # 모든 랜드마크가 탐지되었는지 확인
            if all(key in result for key in required_landmarks):
                print("All landmarks detected.")
                break  # 모든 랜드마크가 탐지되었으므로 루프 종료
            else:
                print("Not all landmarks detected, regenerating image...")
        else:
            print("No detections, regenerating image...")

    # Step 6: 결과 수집 및 Firebase에 저장
    # 데이터베이스에서 전체 데이터 가져오기
    data = get_database_data()

    # 결과 딕셔너리를 저장할 리스트 준비
    for (img_filename, result) in results:
        # 로컬 이미지 경로 구성
        local_img_path = os.path.join(output_dir, img_filename)
        # 다음 이미지 인덱스를 가져오고 'counter'의 'count' 값을 증가시킴
        next_image_index = get_next_image_index(data)
        # Firebase Storage에 저장될 이미지 이름 생성
        key = "def" if False else "gen"
        storage_img_name = f"{key}_{next_image_index:05d}.jpg"

        # 이미지 Firebase Storage에 업로드
        upload_image_to_firebase(local_img_path, storage_img_name)

        # 데이터에 결과 저장 ('gen' 또는 'def' 딕셔너리에 추가)
        save_data_to_firebase(data, storage_img_name, result, is_def=False)

    # 업데이트된 데이터베이스를 Firebase에 저장
    update_database_data(data)

    # 최종 결과 반환
    return Response(status_code=204)
