from .fan import Fan


class HVACController:
    def __init__(self):
        self.fan = Fan(pin=18)
        self.is_active = False

    def update(self, doll_count):
        """인형 수에 따른 풍량 결정"""
        if doll_count == 0:
            self.fan.stop()
            status = "IDLE"
        elif doll_count == 1:
            self.fan.set_speed(0.5)  # 50% 풍량
            status = "LOW"
        else:
            self.fan.set_speed(1.0)  # 100% 풍량
            status = "MAX"

        return status, self.fan.get_speed()

    def shutdown(self):
        self.fan.stop()