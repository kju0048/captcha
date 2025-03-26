// popup.js

// 전역 변수: 캡차 시작 시간 저장
let captchaStartTime = 0;

// 수정된 openAuthPopup 함수: 좌표 표시 완료 후 타이머 시작
async function openAuthPopup() {
    setupColorMapping(); // 색상-숫자 매핑 설정
    document.getElementById('authPopup').style.display = 'flex';
    displayRandomInstruction(); // 랜덤 문구 표시

    // 이미지와 좌표가 모두 로딩 및 표시될 때까지 대기
    await loadRandomImages();
    
    // 이제 캡차 시작 시간 기록
    captchaStartTime = Date.now();
}

// 팝업 닫기
function closeAuthPopup() {
    document.getElementById('authPopup').style.display = 'none';
}

// 인증 결과에 따른 최종 처리 공통 함수
// isSuccess: true이면 인증 성공, false이면 인증 실패
async function finalizeImageResult(isSuccess) {
    const storageRef = firebase.storage();
    const databaseRef = firebase.database().ref(); // 전체 DB 참조
    const folderPath = 'gs://aifront-a7a19.appspot.com/image2';

    try {
        // 1. realtime database에서 전체 데이터 읽기
        const jsonSnapshot = await databaseRef.once('value');
        const jsonData = jsonSnapshot.val();

        if (!jsonData || !jsonData.gen) {
            console.error('JSON 데이터(gen)를 찾을 수 없습니다.');
            return;
        }

        // 2. "gen" 데이터는 객체 형태로 저장되어 있으므로, 해당 이미지 데이터를 찾음
        // 예: imageName2가 "gen_00000.jpg"라면, DB 키는 "gen_00000_jpg"
        const targetKey = imageName2.replace('.jpg', '_jpg');
        let matchingKey = null;
        let matchingData = null;
        for (const [key, data] of Object.entries(jsonData.gen)) {
            if (key === targetKey) {
                matchingKey = key;
                matchingData = data;
                break;
            }
        }
        if (!matchingData) {
            console.error(`이미지 ${imageName2}에 해당하는 JSON 데이터를 찾을 수 없습니다.`);
            return;
        }

        // 3. 인증 결과에 따라 count 업데이트  
        // 인증 성공이면 correct_count 증가, 실패이면 incorrect_count 증가
        if (isSuccess) {
            matchingData.z_correct_count = (matchingData.z_correct_count !== undefined ? matchingData.z_correct_count : 0) + 1;
        } else {
            matchingData.z_incorrect_count = (matchingData.z_incorrect_count !== undefined ? matchingData.z_incorrect_count : 0) + 1;
        }
        // 두 경우 모두 z_all_count 업데이트 (예제 코드 유지)
        matchingData.z_all_count = (matchingData.z_all_count !== undefined ? matchingData.z_all_count : 0) + 1; // 테스트용 임시 코드
        matchingData.z_all_count = (matchingData.z_all_count !== undefined ? matchingData.z_all_count : 0) - 1;
        console.log(`업데이트 후 ${matchingKey}: z_correct_count=${matchingData.z_correct_count}, z_incorrect_count=${matchingData.z_incorrect_count}, z_all_count=${matchingData.z_all_count}`);

        // 4. 만약 z_all_count가 0이면 최종 판단 처리
        if (matchingData.z_all_count === 0) {
            if (matchingData.z_correct_count > matchingData.z_incorrect_count) {
                // 최종 조건 충족: 성공 → 데이터 이동(gen -> def) 및 storage 이미지 이동 (image2 -> image1)
                if (!jsonData.def) {
                    jsonData.def = {};
                }
                jsonData.def[matchingKey] = matchingData;
                delete jsonData.gen[matchingKey];
                console.log(`최종 처리: ${matchingKey}가 def로 이동되었습니다.`);

                await moveImageToFolder(imageName2, 'image2', 'image1');
                console.log(`최종 처리: 이미지 ${imageName2}가 image2에서 image1로 이동되었습니다.`);
            } else {
                // 최종 조건 미충족: 데이터 삭제 및 storage 이미지 삭제
                const imageRef = storageRef.refFromURL(`${folderPath}/${imageName2}`);
                await imageRef.delete();
                delete jsonData.gen[matchingKey];
                console.log(`최종 처리: 이미지 ${imageName2}와 데이터 ${matchingKey}가 삭제되었습니다.`);
            }
        }
        // 5. realtime database 업데이트 (전체 데이터 덮어쓰기)
        await databaseRef.set(jsonData);
    } catch (error) {
        console.error(`작업 중 오류 발생: ${error.message}`);
    }
}

async function handleSubmit() {
    const inputField = document.getElementById('inputField');
    const inputValue = parseInt(inputField.value);
    const instruction = document.getElementById('instructionText').innerText;
    const correctAnswer = calculateMultiplication(instruction);
    const isSuccess = (inputValue === correctAnswer);

    // 캡차 해결 시간 측정 (초 단위, 소수점 2자리까지)
    const captchaEndTime = Date.now();
    const timeTaken = ((captchaEndTime - captchaStartTime) / 1000).toFixed(2);

    closeAuthPopup();

    // 인증 결과와 캡차 해결 시간을 alert로 표시
    if (isSuccess) {
        alert("Success\nTime: " + timeTaken + "s");
    } else {
        alert("Fail\nTime: " + timeTaken + "s");
    }

    // 먼저 인증 결과에 따른 DB 및 이미지 처리 실행
    await finalizeImageResult(isSuccess);

    inputField.value = '';

    // 인증 성공 시, 캡차 시간 정보를 저장하고 survey.html로 이동
    if (isSuccess) {
        localStorage.setItem('captchaTime', timeTaken);
        window.location.href = "survey.html";  // 경로는 실제 파일 위치에 따라 조정
    }

    // REST API GET 요청
    try {
        const response = await fetch('localhost:8000/run');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        console.log('API 호출 완료');
    } catch (error) {
        console.error('API 요청 중 오류 발생:', error);
    }
}


window.openAuthPopup = openAuthPopup;
window.closeAuthPopup = closeAuthPopup;
window.handleSubmit = handleSubmit;
