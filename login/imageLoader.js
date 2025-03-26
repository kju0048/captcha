// 랜덤 이미지 불러오기
// 수정된 loadRandomImages 함수: 좌표 표시가 완료될 때까지 대기합니다.
async function loadRandomImages() {
    const storageRef = firebase.storage().ref();

    // 첫 번째 이미지 로드 및 좌표 처리
    const folderRef1 = storageRef.child('image1');
    const imageList1 = await folderRef1.listAll();
    const randomIndex1 = Math.floor(Math.random() * imageList1.items.length);
    const randomImageRef1 = imageList1.items[randomIndex1];
    const url1 = await randomImageRef1.getDownloadURL();
    imageName1 = randomImageRef1.name;
    document.getElementById('randomImage1').src = url1;
    const fetchPromise1 = fetchCoordinates(imageName1, 'image1-values', 'image1');

    // 두 번째 이미지 로드 및 좌표 처리
    const folderRef2 = storageRef.child('image2');
    const imageList2 = await folderRef2.listAll();
    const randomIndex2 = Math.floor(Math.random() * imageList2.items.length);
    const randomImageRef2 = imageList2.items[randomIndex2];
    const url2 = await randomImageRef2.getDownloadURL();
    imageName2 = randomImageRef2.name;
    document.getElementById('randomImage2').src = url2;
    const fetchPromise2 = fetchCoordinates(imageName2, 'image2-values', 'image2');

    // 두 이미지의 좌표 표시가 모두 완료될 때까지 대기
    await Promise.all([fetchPromise1, fetchPromise2]);
}


function fetchCoordinates(imageId, targetElementId, imageKey) {
    return firebase.database().ref().once('value')
        .then((snapshot) => {
            if (snapshot.exists()) {
                const coordinatesObject = snapshot.val();
                let coordinatesData;
                // 파일명 -> DB key 변환: .jpg -> _jpg
                const dbKey = imageId.replace('.jpg', '_jpg');

                if (imageKey === 'image1') {
                    coordinatesData = coordinatesObject.def?.[dbKey];
                } else if (imageKey === 'image2') {
                    coordinatesData = coordinatesObject.gen?.[dbKey];
                }

                if (coordinatesData) {
                    displayRandomValues(coordinatesData, targetElementId, imageKey);
                } else {
                    console.log(`DB에서 해당 key(${dbKey})에 해당하는 좌표 데이터를 찾지 못했습니다.`);
                }
            }
        })
        .catch((error) => {
            console.error(`Error fetching coordinates: ${error}`);
        });
}

// 이미지 및 좌표 이동 함수
async function moveImageToFolder(imageName, sourceFolder, destinationFolder) {
    console.log("이미지 및 좌표 이동 함수 시작");

    // Firebase 스토리지 참조 생성
    const storageRef = firebase.storage();

    // 원본 폴더 참조 (gs:// 경로 사용)
    const folderRef = storageRef.refFromURL(`gs://aifront-a7a19.appspot.com/${sourceFolder}`);
    console.log("원본 폴더 참조: ", folderRef.fullPath);

    try {
        // 폴더 내 파일 리스트 가져오기
        const imageList = await folderRef.listAll();
        console.log("원본 폴더 내 파일 목록: ", imageList);

        // 파일 이름으로 해당 파일 참조
        const fileRef = imageList.items.find(item => item.name === imageName);

        if (!fileRef) {
            throw new Error(`파일 '${imageName}'을(를) ${sourceFolder}에서 찾을 수 없습니다.`);
        }

        console.log("파일 참조: ", fileRef.fullPath);

        // 파일 URL 가져오기
        const url = await fileRef.getDownloadURL();
        console.log("파일 URL: ", url);

        // 파일 다운로드
        const response = await fetch(url);
        const blob = await response.blob();

        // 대상 폴더 참조 (gs:// 경로 사용)
        const destinationRef = storageRef.refFromURL(`gs://aifront-a7a19.appspot.com/${destinationFolder}/${imageName}`);
        console.log("대상 폴더 참조: ", destinationRef.fullPath);

        // 파일 업로드
        await destinationRef.put(blob);
        console.log(`파일이 ${destinationFolder}에 업로드되었습니다.`);

        // 원본 파일 삭제
        await fileRef.delete();
        console.log(`원본 파일 '${imageName}'가 삭제되었습니다.`);

        console.log(`파일 '${imageName}'가 '${sourceFolder}'에서 '${destinationFolder}'로 이동되었습니다.`);
    } catch (error) {
        console.error(`파일 이동 중 오류 발생: ${error.message}`);
    }
}




function displayRandomValues(coordinatesData, targetElementId, imageKey) {
    const targetElement = document.getElementById(targetElementId);
    targetElement.innerHTML = '';

    const points = [
        { key: "lefteye", x: coordinatesData.lefteye_x, y: coordinatesData.lefteye_y },
        { key: "righteye", x: coordinatesData.righteye_x, y: coordinatesData.righteye_y },
        { key: "nose", x: coordinatesData.nose_x, y: coordinatesData.nose_y },
        { key: "leftmouth", x: coordinatesData.leftmouth_x, y: coordinatesData.leftmouth_y },
        { key: "rightmouth", x: coordinatesData.rightmouth_x, y: coordinatesData.rightmouth_y }
    ];

    // 이미지 박스와 원본 이미지의 크기 비율 계산
    const imageBox = targetElement.closest('.image-box');
    const imageElement = imageBox.querySelector('img');
    const originalWidth = 128; // 원본 이미지 너비
    const originalHeight = 128; // 원본 이미지 높이
    const currentWidth = imageElement.offsetWidth;
    const currentHeight = imageElement.offsetHeight;

    const widthRatio = currentWidth / originalWidth;
    const heightRatio = currentHeight / originalHeight;

    const shuffledColors = [...colors].sort(() => Math.random() - 0.5);
    points.forEach((coord, index) => {
        const color = shuffledColors[index % shuffledColors.length];
        const value = colorMapping[color];

        const circle = document.createElement('div');
        circle.className = 'random-color-circle';
        circle.style.backgroundColor = getColorCode(color);

        // 좌표 변환
        const adjustedX = coord.x * widthRatio;
        const adjustedY = coord.y * heightRatio;

        circle.style.top = `${adjustedY - 10}px`; // 원 크기의 절반을 빼서 중앙에 위치
        circle.style.left = `${adjustedX - 10}px`;

        targetElement.appendChild(circle);

        randomValues[imageKey][coord.key] = value;
    });
}
