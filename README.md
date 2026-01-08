# HERO - 보이는 인사관리,통합 HR 대시보드

<div align="center">
  <img
    width="300"
    height="200"
    alt="logo"
    src="https://github.com/user-attachments/assets/d265cec7-0b13-45a8-88f0-68e6dc84a432"
  />
</div>

<hr>

## 목차
#### [팀 소개](#팀-소개-1)
#### [프로젝트 소개](#프로젝트-소개-1)
#### [주요 기능](#주요-기능-1)
#### [기술 스택](#기술-스택-1)
#### [시스템 아키텍처](#시스템-아키텍처-1)
#### [WBS](#wbs-1)
#### [요구사항 명세서](#요구사항-명세서-1)
#### [DDD](#ddd-1)
#### [ERD](#erd-1)
#### [Wire Frame](#wire-frame-1)
#### [API 명세서](#api-명세서-1)
#### [단위 테스트](#단위-테스트-1)

<hr>

## 팀 소개

<br>
<div align="center">
  <table>
  <tr>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/4ee8a86c-3f8a-4a64-bdfa-7ea92219c7f6" />
    </td>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/7ecc1ad8-c44a-4c4f-96c2-ce68e0a14ed5" />
    </td>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/61d2af39-81fd-4cc8-956c-5a5550c609dd" />
    </td>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/0a339929-90bb-4600-8476-206d1cf52346" />
    </td>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/75176e7f-bafc-4e43-a559-36c677395664" />
    </td>
    <td align="center">
      <img width="192" height="322" alt="image" src="https://github.com/user-attachments/assets/3c5ab666-965b-4e7c-b184-719b8b8ba41e" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/dddd0ng"><b>곽동근</b></a>
    </td>
    <td align="center">
      <a href="https://github.com/indy0322"><b>김승민</b></a>
    </td>
    <td align="center">
      <a href="https://github.com/bynmch"><b>변민철</b></a>
    </td>
    <td align="center">
      <a href="https://github.com/Seung-Geon"><b>이승건</b></a>
    </td>
    <td align="center">
      <a href="https://github.com/Easy-going12"><b>이지윤</b></a>
    </td>
    <td align="center">
      <a href="https://github.com/haenin"><b>최혜원</b></a>
    </td>
  </tr>
</table>
</div>
<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<br>

<hr>

## 프로젝트 소개
**HERO** 는 기업 내에 분산되어 있던 **근태, 휴가, 평가, 결재 데이터**를 하나의 흐름으로 통합 관리하는 **인사(HR) 시스템**입니다.

각각 독립적으로 운영되던 인사 데이터를 유기적으로 연결함으로써
**데이터 취합 및 관리에 소요되던 시간을 획기적으로 줄이고**, <br>
핵심 인사 지표를 **대시보드로 한눈에 파악**할 수 있도록 설계되었습니다.

또한,
**역할(Role) 기반 권한 관리**를 통해
관리자, 부서장, 일반 사원 등 사용자 유형에 따라 **필요한 기능만 노출되는 UI**를 제공하여 <br>
보안성과 사용성을 동시에 고려한 인사 운영 환경을 구현합니다.

HERO는 단순한 기능 나열이 아닌,
**인사 데이터 간의 시너지를 극대화하여 효율적인 조직 운영을 지원하는 HR 플랫폼을 목표**로 합니다.

<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>


## 주요 기능
### 조직 운영을 고려한 UX
- 개인 근태 이력, 부서별 휴가 일정, 알림 설정 등을 사용자가 이해하기 쉽도록 디자인 하였습니다.

### 대시보드를 이용한 시각화
- 인건비, 사원 평가, 퇴사 데이터 등을 대시보드를 사용하여 시각화시켰습니다.
  
### AI를 사용한 업무 공수 절감
- 평가 가이드 라인 위반 여부, 우수 사원 조사 등의 번거로운 업무를 AI를 사용하여 작업 시간을 절감하였습니다.

### 사용자별 권한 세분
- 권한에 따른 접근 페이지를 제한함으로써 보안적 측면을 강화화였습니다.

<hr>

## 기술 스택
### 🧩 Backend - Spring
![Java](https://img.shields.io/badge/java-007396?style=for-the-badge&logo=java&logoColor=white)
![Spring Boot](https://img.shields.io/badge/spring%20boot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white)
![Spring Security](https://img.shields.io/badge/spring%20security-6DB33F?style=for-the-badge&logo=springsecurity&logoColor=white)
![Spring Data JPA](https://img.shields.io/badge/spring%20data%20jpa-6DB33F?style=for-the-badge)
![JWT](https://img.shields.io/badge/jwt-000000?style=for-the-badge)
![Lombok](https://img.shields.io/badge/lombok-BC4521?style=for-the-badge)
![Gradle](https://img.shields.io/badge/gradle-02303A?style=for-the-badge&logo=gradle&logoColor=white)
![Swagger](https://img.shields.io/badge/swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=white)
![WebSocket](https://img.shields.io/badge/websocket-000000?style=for-the-badge&logo=socketdotio&logoColor=white)


### 🤖 Backend - Python
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)

### 🎨 Frontend
![Vue.js](https://img.shields.io/badge/vue.js-4FC08D?style=for-the-badge&logo=vuedotjs&logoColor=white)
![Pinia](https://img.shields.io/badge/pinia-FFD859?style=for-the-badge)
![Vue Router](https://img.shields.io/badge/vue%20router-4FC08D?style=for-the-badge)
![Vite](https://img.shields.io/badge/vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Axios](https://img.shields.io/badge/axios-5A29E4?style=for-the-badge)
![TypeScript](https://img.shields.io/badge/typescript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Chart.js](https://img.shields.io/badge/chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### 🗄️ Database
![MariaDB](https://img.shields.io/badge/mariadb-003545?style=for-the-badge&logo=mariadb&logoColor=white)
![Amazon RDS](https://img.shields.io/badge/amazon%20rds-527FFF?style=for-the-badge&logo=amazonrds&logoColor=white)

### ☁️ Cloud / Infrastructure
![AWS](https://img.shields.io/badge/aws-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![AWS IAM](https://img.shields.io/badge/aws%20iam-FF9900?style=for-the-badge)
![Amazon S3](https://img.shields.io/badge/amazon%20s3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)


### 🔄 CI / CD 
![GitHub Actions](https://img.shields.io/badge/github%20actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)


### 🛠 Tools
![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github)
![Notion](https://img.shields.io/badge/notion-000000?style=for-the-badge&logo=notion)
![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)
![Figma](https://img.shields.io/badge/figma-F24E1E?style=for-the-badge&logo=figma&logoColor=white)


<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>

## 시스템 아키텍처
<img width="2121" height="1262" alt="KakaoTalk_Photo_2026-01-05-16-59-19" src="https://github.com/user-attachments/assets/db665141-4fce-48d2-806f-c887f7e971c5" />


<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>

## WBS
<img width="1241" height="581" alt="스크린샷 2026-01-08 오전 10 35 48" src="https://github.com/user-attachments/assets/b992c4e2-6c2f-4a3c-8207-f8cfc5bbb3f6" />

<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>

## 요구사항 명세서
<img width="845" height="630" alt="스크린샷 2026-01-08 오후 1 48 17" src="https://github.com/user-attachments/assets/8ed93daf-634e-4b8e-b71d-9be077c9c43b" />
<img width="845" height="630" alt="스크린샷 2026-01-08 오후 1 48 40" src="https://github.com/user-attachments/assets/23e94920-cf4e-49ea-8f3b-d0a2b9da3f60" />
<img width="845" height="571" alt="스크린샷 2026-01-08 오후 1 48 53" src="https://github.com/user-attachments/assets/de2bb6e4-ac83-4e5d-86af-5b65646e4480" />

<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>

## DDD
<img width="1190" height="643" alt="스크린샷 2026-01-05 오후 5 06 50" src="https://github.com/user-attachments/assets/46fb17fe-65dd-4aa6-b901-7007a671e5b9" />
<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>

## ERD
<img width="1302" height="743" alt="스크린샷 2026-01-05 오후 4 44 13" src="https://github.com/user-attachments/assets/8958b9d9-7e29-4254-b8ed-522a6ca3f700" />
<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>


## Wire Frame
<details>
  <summary>Wire Frame</summary>
<details>
  <summary>로그인</summary>
<img width="1213" height="346" alt="image" src="https://github.com/user-attachments/assets/252531b9-16df-4056-8a91-0b225bda94b0" />
</details>
<details>
  <summary>근태 관리</summary>
<img width="1102" height="815" alt="image" src="https://github.com/user-attachments/assets/604003cd-3daf-47dd-8fd0-4d30c82cffb5" />
</details>
<details>
  <summary>휴가 / 연차</summary>
<img width="1147" height="717" alt="image" src="https://github.com/user-attachments/assets/2e57414d-64c5-4a7d-892c-27cb421bc64e" />
</details>
<details>
  <summary>전자 결재</summary>
<img width="1137" height="786" alt="image" src="https://github.com/user-attachments/assets/fed47ffc-ab2c-4bf7-b696-bfc81fad9bcd" />
</details>
<details>
  <summary>성과 평가</summary>
<img width="1045" height="827" alt="image" src="https://github.com/user-attachments/assets/36a5087e-4073-465b-8f52-c52358323e62" />
<img width="608" height="762" alt="image" src="https://github.com/user-attachments/assets/fc6947e4-061f-418f-b668-a5f88fc58890" />
</details>
<details>
  <summary>급여</summary>
<img width="888" height="598" alt="image" src="https://github.com/user-attachments/assets/2f75eac6-3773-4457-b761-652aefaecd23" />
</details>
<details>
  <summary>급여 관리</summary>
<img width="1206" height="733" alt="image" src="https://github.com/user-attachments/assets/8257b82f-e08d-433d-ba72-b54ab7fe1e53" />
<img width="420" height="831" alt="image" src="https://github.com/user-attachments/assets/dcc2f6b6-71c8-465e-889b-0e613b596701" />
</details>
<details>
  <summary>사원 관리</summary>
<img width="652" height="802" alt="image" src="https://github.com/user-attachments/assets/ecf0407c-7387-46b9-ac4e-06b3a3fbbc0d" />
</details>
<details>
  <summary>조직도</summary>
<img width="1100" height="410" alt="image" src="https://github.com/user-attachments/assets/afff67ec-b6e1-427c-a73d-95328d197936" />
</details>
<details>
  <summary>알림</summary>
<img width="1027" height="667" alt="image" src="https://github.com/user-attachments/assets/d3ed9362-45c5-4c9f-9754-f68f59d8762b" />
</details>
<details>
  <summary>설정</summary>
<img width="943" height="762" alt="image" src="https://github.com/user-attachments/assets/3a5dea74-5e36-4062-8773-4590977bedb7" />
</details>
</details>
<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<br>
<hr>


## API 명세서
<details>
  <summary>REST API</summary>
<details>
  <summary>급여 수당 마스터 API</summary>
<img width="1301" height="328" alt="image" src="https://github.com/user-attachments/assets/03b17f62-a15f-4f98-acfb-0ec021b0b8c3" />
</details>
<details>
  <summary>퇴직 관리 API</summary>
<img width="1301" height="399" alt="image" src="https://github.com/user-attachments/assets/a3ab9d5e-a1fc-4a75-be55-566369603630" />
</details>
<details>
  <summary>알림 설정 API</summary>
<img width="1301" height="165" alt="image" src="https://github.com/user-attachments/assets/f6333a7e-ec52-4c3f-9d8c-ee48ddfc34a8" />
</details>
<details>
  <summary>내 급여 리포트 API</summary>
<img width="1301" height="165" alt="image" src="https://github.com/user-attachments/assets/e9aa76f4-36d3-4483-8ef4-02353702c732" />
</details>
<details>
  <summary>휴가 API</summary>
<img width="1301" height="207" alt="image" src="https://github.com/user-attachments/assets/6050d313-a929-4be5-a270-98aa72473023" />
</details>
<details>
  <summary>급여 배치 API</summary>
<img width="1301" height="466" alt="image" src="https://github.com/user-attachments/assets/c1e1c74f-416e-432f-8366-6868f5e6094d" />
</details>
<details>
  <summary>급여 정책 설정 참조 API</summary>
<img width="1301" height="155" alt="image" src="https://github.com/user-attachments/assets/f3768b6a-642a-436f-9cb4-a3a76944ccdf" />
</details>
<details>
  <summary>홈 대시보드 API</summary>
<img width="1301" height="462" alt="image" src="https://github.com/user-attachments/assets/8cc23109-f072-495f-ae57-dcb4e4b5d137" />
</details>
<details>
  <summary>급여 조회 API</summary>
<img width="1301" height="159" alt="image" src="https://github.com/user-attachments/assets/35f20297-0c8f-474b-8a22-b1d70a5fd7fd" />
</details>
<details>
  <summary>급여 분석 API</summary>
<img width="1301" height="211" alt="image" src="https://github.com/user-attachments/assets/a83eac1c-4da7-4e82-8acd-aff314d27fbc" />
</details>
<details>
  <summary>급여 계좌 API</summary>
<img width="1301" height="309" alt="image" src="https://github.com/user-attachments/assets/403f947d-ba71-405c-8838-8e6dca68c775" />
</details>
<details>
  <summary>급여 정책 관리 API</summary>
<img width="1301" height="596" alt="image" src="https://github.com/user-attachments/assets/09681d08-ca8e-4dff-9a1c-d314929e5bf9" />
<img width="1301" height="304" alt="image" src="https://github.com/user-attachments/assets/b376564f-f99b-4a3f-9122-7fff45ac9c5b" />
</details>
<details>
  <summary>근태 API</summary>
<img width="1301" height="548" alt="image" src="https://github.com/user-attachments/assets/7a33639e-0ce3-4d91-98e5-3bfd300ccc9e" />
</details>
<details>
  <summary>급여 공제 마스터 API</summary>
<img width="1301" height="302" alt="image" src="https://github.com/user-attachments/assets/eec6e7c7-2ac5-48b9-b03a-e8612cb43fa1" />
</details>
<details>
  <summary>파이썬 연동 API</summary>
<img width="1301" height="218" alt="image" src="https://github.com/user-attachments/assets/54116da4-1c10-4e97-a7fa-ba6b4bc8bf23" />
</details>
<details>
  <summary>알림 API</summary>
<img width="1301" height="454" alt="image" src="https://github.com/user-attachments/assets/f1802456-81e1-49b5-a541-a857cfca8b13" />
</details>
<details>
  <summary>급여 명세서 API</summary>
<img width="1301" height="112" alt="image" src="https://github.com/user-attachments/assets/7069b5a5-36c6-47cd-9e58-0314368ff3d6" />
</details>
<details>
  <summary>평가 템플릿 API</summary>
<img width="1301" height="689" alt="image" src="https://github.com/user-attachments/assets/b9efcecb-c52a-4fc8-9400-3bb5a8c985f3" />
<img width="1301" height="550" alt="image" src="https://github.com/user-attachments/assets/022e2734-b0d8-4eaa-990c-83e0b880f753" />
</details>
<details>
  <summary>직원 API</summary>
<img width="899" height="688" alt="image" src="https://github.com/user-attachments/assets/f552bd6e-3a7e-473d-9cdc-193f08a46870" />
</details>
<details>
  <summary>사원 패스워드 API</summary>
<img width="1797" height="289" alt="image" src="https://github.com/user-attachments/assets/f1111c46-55dd-4d1e-a22d-2568318358ee" />
</details>
<details>
  <summary>인증 API</summary>
<img width="1804" height="297" alt="image" src="https://github.com/user-attachments/assets/13d8d6da-df16-44c7-8938-ed2eb0562681" />
</details>
<details>
  <summary>결재 API</summary>
<img width="1301" height="745" alt="image" src="https://github.com/user-attachments/assets/28f44dff-c672-459e-8c33-75a50d1ea3b1" />
</details>
<details>
  <summary>환경설정 API</summary>
<img width="895" height="920" alt="image" src="https://github.com/user-attachments/assets/2bc22fef-2af5-4c2c-be28-c7926b9323cc" />
</details>
<details>
  <summary>퇴직 API</summary>
<img width="897" height="309" alt="image" src="https://github.com/user-attachments/assets/6414793b-60d8-40b9-8796-a08add614b58" />
</details>
<details>
  <summary>승진 API</summary>
<img width="898" height="535" alt="image" src="https://github.com/user-attachments/assets/5284a8e2-63d4-4316-b8b0-6444c7bdd0d7" />
</details>
<details>
  <summary>조직도 API</summary>
<img width="897" height="194" alt="image" src="https://github.com/user-attachments/assets/5cdebc31-8db5-42e8-9ab9-c5c8a4c0abec" />
</details>

</details>
<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<br>
<hr>



## 단위 테스트

<details>
  <summary>테스트 목록</summary>
<details>
  <summary>근태 관리 단위 테스트</summary>
  <details>
    <summary>근태 조회</summary>
    <img width="1236" height="490" alt="image" src="https://github.com/user-attachments/assets/d1542966-fddb-4e5b-9a61-450cee6c3267" />
  </details>
  <details>
    <summary>근태 이벤트</summary>
    <img width="1556" height="617" alt="image" src="https://github.com/user-attachments/assets/c7f53848-784f-4ffa-b6dd-f0904cbc9d1b" />
  </details>
</details>
<details>
  <summary>휴가/연차 단위 테스트</summary>
<img width="1301" height="399" alt="image" src="" />
</details>
<details>
  <summary>전자 결재 단위 테스트</summary>
<img width="1296" height="593" alt="image" src="https://github.com/user-attachments/assets/22ecd0a5-6f73-4710-b9c2-32135926af6f" />
</details>
<details>
  <summary>성과 평가 단위 테스트</summary>
<img width="1436" height="895" alt="image" src="https://github.com/user-attachments/assets/ed5b7825-ebe4-4c8a-b605-7aceed00014c" />
</details>
<details>
  <summary>급여 관리 단위 테스트</summary>
<img width="1301" height="165" alt="image" src="" />
</details>
<details>
  <summary>조직도 단위 테스트</summary>
<img width="1125" height="596" alt="image" src="https://github.com/user-attachments/assets/6a1575c6-8215-4bf7-8b74-67ac866fa683" />
</details>
<details>
  <summary>사원 관리 단위 테스트</summary>
  <details>
    <summary>사원 CUD</summary>
    <img width="1269" height="582" alt="image" src="https://github.com/user-attachments/assets/cfeddccf-8706-4d45-b8b0-318eaa6315a5" />
  </details>
  <details>
    <summary>사원 비밀번호 변경</summary>
    <img width="1315" height="572" alt="image" src="https://github.com/user-attachments/assets/6ca3c450-e611-4a60-b5f4-b88179f6a52b" />
  </details>
  <details>
    <summary>사원 프로필 정보RU</summary>
    <img width="1095" height="499" alt="image" src="https://github.com/user-attachments/assets/213b56da-2a1e-4605-baa0-51deb07a113f" />
  </details>
  <details>
    <summary>사원 정보 조회</summary>
    <img width="940" height="486" alt="image" src="https://github.com/user-attachments/assets/a60dbf96-1244-42a9-a7e2-3ae63d609e2b" />
  </details>
  <details>
    <summary>사원 직인 CUD</summary>
    <img width="1000" height="525" alt="image" src="https://github.com/user-attachments/assets/1c406a10-9d53-4318-90f9-596ffd1a2758" />
  </details>
  <details>
    <summary>승진 관련 조회</summary>
    <img width="1167" height="526" alt="image" src="https://github.com/user-attachments/assets/c2d723be-569b-4ea1-9567-6a15897fa9ef" />
  </details>
  <details>
    <summary>승진 관련 CU</summary>
    <img width="1019" height="530" alt="image" src="https://github.com/user-attachments/assets/ed509107-56b8-46aa-a6d3-9b94ce12cf88" />
  </details>
  <details>
    <summary>승진 처리</summary>
    <img width="1202" height="526" alt="image" src="https://github.com/user-attachments/assets/f919b032-f851-4b3b-b31e-3caf2c3b2689" />
  </details>
  <details>
    <summary>퇴사 현황 조회</summary>
    <img width="1043" height="489" alt="image" src="https://github.com/user-attachments/assets/5a6e1f96-9260-4ee2-88bc-35fbcc2ec581" />
  </details>
  
</details>
<details>
  <summary>알림 단위 테스트</summary>
<img width="1301" height="165" alt="image" src="" />
</details>
<details>
  <summary>파이썬 서버 연동 단위 테스</summary>
<img width="1196" height="622" alt="image" src="https://github.com/user-attachments/assets/086b21ef-4534-4984-b419-b4d624b11798" />
</details>
</details>


<br>
<div align="right">
  <a href="#목차">🔝 맨 위로</a>
</div>
<hr>
