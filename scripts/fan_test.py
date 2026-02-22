from gpiozero import LED
from time import sleep

fan = LED(18)

print("팬 테스트 시작 (3초 간격 ON/OFF)")
try:
    while True:
        print("팬 가동")
        fan.on()
        sleep(3)

        print("팬 정지")
        fan.off()
        sleep(3)
except KeyboardInterrupt:
    fan.off()
    print("테스트 종료")