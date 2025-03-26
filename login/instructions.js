// 텍스트 문구 배열 생성 (영어로 변환)
const instructions = [
    "Compute the product of the left eyes of both images",
    "Compute the product of the right eyes of both images",
    "Compute the product of the noses of both images",
    "Compute the product of the left corners of the mouth of both images",
    "Compute the product of the right corners of the mouth of both images"
];

const coordinateKeys = {
    "Compute the product of the left eyes of both images": "lefteye",
    "Compute the product of the right eyes of both images": "righteye",
    "Compute the product of the noses of both images": "nose",
    "Compute the product of the left corners of the mouth of both images": "leftmouth",
    "Compute the product of the right corners of the mouth of both images": "rightmouth"
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
