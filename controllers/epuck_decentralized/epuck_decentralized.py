from deepbots.robots.controllers.csv_robot import CSVRobot
from controller import Robot

# =============================================================================
# DECENTRALIZED ROBOT CONTROLLER
# Identical to centralized epuck_driver — robot just sends proximity sensors
# and receives motor commands. All intelligence (PPO, pheromone) lives in the
# decentralized supervisor. No changes needed at the robot level.
# =============================================================================

class EpuckDecentralized(CSVRobot):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # 8 IR proximity sensors
        self.ps = []
        for i in range(8):
            sensor = self.getDevice(f'ps{i}')
            sensor.enable(self.time_step)
            self.ps.append(sensor)

        # Camera (enabled for realism; tag detection handled by supervisor)
        self.camera = self.getDevice('camera')
        if self.camera:
            self.camera.enable(self.time_step)

        # Differential drive motors
        self.left_motor  = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def create_message(self):
        """Send 8 normalised proximity sensor readings to supervisor."""
        return [s.getValue() / 4096.0 for s in self.ps]

    def use_message_data(self, message):
        """Receive [left, right] motor commands in [-1, 1] from supervisor."""
        if message:
            try:
                left  = float(message[0])
                right = float(message[1])
                max_speed    = 6.28   # rad/s (e-puck hardware limit)
                scale_factor = 6.0    # maps [-1,1] → [-6.28, 6.28]
                left  = max(min(left  * scale_factor, max_speed), -max_speed)
                right = max(min(right * scale_factor, max_speed), -max_speed)
                self.left_motor.setVelocity(left)
                self.right_motor.setVelocity(right)
            except (ValueError, IndexError):
                pass


if __name__ == '__main__':
    robot = EpuckDecentralized()
    robot.run()
