from gpiozero import PWMLED

class Fan:
    def __init__(self, pin=18):
        # 모스펫 제어를 위해 PWMLED 사용
        self._device = PWMLED(pin)
        self._current_speed = 0.0

    def set_speed(self, speed: float):
        """0.0 ~ 1.0 사이의 속도 설정"""
        # 범위 제한 (0~1 사이로 안전하게)
        speed = max(0.0, min(1.0, speed))
        self._device.value = speed
        self._current_speed = speed

    def stop(self):
        self.set_speed(0.0)

    def get_speed(self):
        return self._current_speed