// 텍스트 문구 배열 생성
const instructions = [
    "두 이미지의 왼쪽 눈의 곱을 구하시오",
    "두 이미지의 오른쪽 눈의 곱을 구하시오",
    "두 이미지의 코의 곱을 구하시오",
    "두 이미지의 왼쪽 입꼬리의 곱을 구하시오",
    "두 이미지의 오른쪽 입꼬리의 곱을 구하시오"
];

const coordinateKeys = {
    "두 이미지의 왼쪽 눈의 곱을 구하시오": "lefteye",
    "두 이미지의 오른쪽 눈의 곱을 구하시오": "righteye",
    "두 이미지의 코의 곱을 구하시오": "nose",
    "두 이미지의 왼쪽 입꼬리의 곱을 구하시오": "leftmouth",
    "두 이미지의 오른쪽 입꼬리의 곱을 구하시오": "rightmouth"
};

// 랜덤 문구 표시 함수
function displayRandomInstruction() {
    const randomIndex = Math.floor(Math.random() * instructions.length); // 랜덤 인덱스 생성
    document.getElementById('instructionText').innerText = instructions[randomIndex]; // 랜덤 문구 설정
}

// 곱셈 연산 수행 함수
function calculateMultiplication(instruction) {
    const key = coordinateKeys[instruction];
    const value1 = randomValues.image1[key];
    const value2 = randomValues.image2[key];
    return value1 * value2;
}
