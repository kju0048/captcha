// 텍스트 문구 배열 생성 (영어로 변환)
const instructions = [
    "Enter the product of the values corresponding to the left eye of the two portraits.",
    "Enter the product of the values corresponding to the right eye of the two portraits.",
    "Enter the product of the values corresponding to the nose of the two portraits.",
    "Enter the product of the values corresponding to the left corner of the mouth of the two portraits.",
    "Enter the product of the values corresponding to the right corner of the mouth of the two portraits."
  ];
  
  const coordinateKeys = {
    "Enter the product of the values corresponding to the left eye of the two portraits.": "lefteye",
    "Enter the product of the values corresponding to the right eye of the two portraits.": "righteye",
    "Enter the product of the values corresponding to the nose of the two portraits.": "nose",
    "Enter the product of the values corresponding to the left corner of the mouth of the two portraits.": "leftmouth",
    "Enter the product of the values corresponding to the right corner of the mouth of the two portraits.": "rightmouth"
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
