// 색상 배열 및 색상-숫자 매핑
const colors = ["yellow", "green", "red", "purple", "blue"];
const colorMapping = {}; // 색상-숫자 매핑 객체

// 색상 이름에 따른 실제 색상 코드 반환
function getColorCode(color) {
    const colorCodes = {
        "yellow": "yellow",
        "green": "green",
        "red": "red",
        "purple": "purple",
        "blue": "blue"
    };
    return colorCodes[color];
}

// 색상-숫자 매칭 및 표시
function setupColorMapping() {
    const colorNumberDisplay = document.getElementById("colorNumberDisplay");
    colorNumberDisplay.innerHTML = ''; // 기존 데이터 초기화

    colors.forEach(color => {
        const randomValue = getRandomValue(); // 랜덤 숫자 생성
        colorMapping[color] = randomValue;

        // 화면에 색상-숫자 매칭 정보 추가
        const span = document.createElement('span');
        span.innerHTML = `
            <div class="color-circle" style="background-color: ${getColorCode(color)};"></div>
            ${color}: ${randomValue}
        `;
        colorNumberDisplay.appendChild(span);
    });
}

// 랜덤 값 생성 로직
const availableValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11];
let usedRandomValues = [];

function getRandomValue() {
    if (usedRandomValues.length >= availableValues.length) {
        usedRandomValues = [];
    }
    let value;
    do {
        const randomIndex = Math.floor(Math.random() * availableValues.length);
        value = availableValues[randomIndex];
    } while (usedRandomValues.includes(value));
    usedRandomValues.push(value);
    return value;
}
