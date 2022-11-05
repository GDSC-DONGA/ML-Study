## Intro

**인공지능 소개 및 발전 과정을 소개**

<br>

## 1. 인공지능 소개

**인공지능이란?** 

* 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현하려는 컴퓨터 과학의 세부분야 중 하나이다.



**관점 4가지**

* **Thinking Humanly** : 사람이 어떻게 사고하는지를 구현해서 사람처럼 생각할 수 있는 인공지능을 만들겠다는 관점
* **Thinking Rationally** : 사람이 어떻게 사고하는지를 구현해서 이성적인 대리인을 만들겠다는 관점
* **Acting Humanly** : 사람이 하는 지능적인 행동을 구현해서 사람처럼 행동할 수 있는 인공지능을 만들겠다는 관점
* **Acting Rationally** : 사람이 하는 지능적인 행동을 구현해서 이성적인 대리인을 만들겠다는 관점



**Thinking Humanly 보다는 Acting Humanly가 상대적으로 쉽다**

* 지능은 정의하기 어렵지만 지능적 행동은 그렇지 않기 때문이다.
* 그리고 `Acting Humanly` 관점에서는 지능적 행동을 할 때 그 안에서 어떤 사고가 진행 되는지는 알 필요가 없기 때문이다.
* EX) `Turing Test`



**The Turing test (튜링)**

* `행동` 관점에서 인공지능의 판단 테스트 예시이다

* 예로 34957+70764?

  * `Acting Rationaly` : (즉답) 105621

  * `Acting Humanly` : (30초 정도 쉬고) 105621



**AI의 주요 구성 요소** : `Turing Test`를 통과하기 위한 능력

* Knowledge Representation : 이미 가진 지식의 표현
* Automated Reasoning : 기존 지식으로부터의 추론
* Learing : 학습
* Language/image Understanding : 조사관의 질문을 이해할 수 있어야 한다
  + Robotics : 주위 환경에 반응



**인공지능에서 중요한 주제**

* `Sensing` :  감각을 이용해서 환경을 이해하고 환경을 측정해내는 것
  * Computer Vision, 음성 인식, 자연어 이해
* `Thinking` : 지식을 표현하는 것, 문제의 해결능력, 계획, 예전의 경험으로부터 새로운 지식을 얻어내는 학습과정 등
  * 지식 표현, 문제 해결능력, 학습
* `Acting` : 생각하고 사고한 내용을 표현하는 방법
  * Robotics, 말하기/언어 합성



**인공지능의 적용 범위(활용도)에 따라 ANI, AGI, ASI로 구분**

* `ANI(Artificial Narrow Intelligence)` : '좁은인공지능' or '약인공지능'
  * 게임, 의학, 법학 등 특정 분야의 제한된 소재를 다루는 인공지능
  * 대표적인 ANI : 알파고(바둑), IBM 왓슨(자연어)
* `AGI(Artificial General Intelligence)` : '범용인공지능' or '강인공지능'
  * 실험적으로 시도되고 있는 분야
  * ANI가 특정 분야에 적용되는 인공지능이라면 AGI는 그 적용 분야가 넓어진다.
* `ASI(Artificial Super Intelligence)` : '초인공지능'
  * 아직 구현되지 않은 분야
  * AGI에 비해 더 강력한 지능체계를 갖추고 있어서 스스로 목표를 설정하여 지식을 강화하는 인공지능

<br>

## 2. 발전 과정

**The Mechanical Turk(1770)**

* 84년 동안이나 인공지능인 줄 알았으나 거짓말이였다.
  * 사람이 조작하고 있었음

**The Turing Test(1950)**

* 위에서 언급했던 `Turing Test`이다.
* Alan Turing은 `Turing Test`를 ‘imitation game’(모방 게임) 이라고 이름을 붙임

**Speech recognition 레커그니션 (1952) 음성인식**

* Bell Lab에서 개발한 Audrey (오드리) system으로 전화번호 자동 인식기 개발
  * Shoebox 음성인식 시스템 : 16단어를 인식
  * Harpy system 음성인식 시스템 : 1000개 정도의 단어를 인식

**The birth of AI(1956) : A Workshop at Dartmouth**

* 인공지능이라는 학문이 처음으로 태동하게 된 일종의 `학술회의`

- Computer chess(컴퓨터 체스) : Arthur Samuel(아서 사무엘)이 `alpha-beta pruning`이라는 알고리즘을 제안
  * 당시 아마추어 레벨까지 도달

**Paper 'Checkers was solved'(2007) by Jonathan Schaeffer**

* 체스라는 게임을 수학적으로 완전히 분석한 첫 번째 논문

**Shakey(1966-1972)**

* 최초의 이동로봇

**DeepBlue(1997)**

* 체스 프로그램
* 1997년, 당시 챔피언(인간)을 3.5:2.5로 AI가 승리

**Honda's Asimo(2004)**

* 사람과 매우 유사한 방식으로 걸을 수 있는 휴머노이드 로봇

**DARPA Urban Challenge(2007)**

* `DARPA` : 미국 국방부의 펀딩 에이전시
* 실제 도시처럼 만들어놓은 환경에서 자율주행 자동차가 얼마큼 잘 이동할 수 있는지를 경진대회 형태로 참여
* 근래 자율주행 자동차라는 화두의 시초가 된 챌린지
* 챌린지 관련자가 현재 자율주행 연구의 핵심 인력으로 활약

**Watson by IBM(2011)**

* 퀴즈쇼에서 AI가 사람을 이김

**AlphaGo by Google DeepMind(2016)**

* 실제 크기의 바둑판에서 처음으로 프로 기사들을 이긴 프로그램
* 2016년 이세돌과의 대회에서 4:1로 AI(알파고)가 승리

**AlphaGo Zero(2017)**

* 인간의 데이터가 아닌, 스스로 학습하는 방식으로 만들어진 인공지능(learning from scratch) 

* 2017년 커제와의 대회에서 3:0으로 승리

**AlphaZero**

* 알파고 제로를 만든지 한달 반 만에 알파제로를 만듬
* 바둑뿐만 아니라, 체스나 장기까지 학습 가능

