from deepbots.robots.controllers.csv_robot import CSVRobot
from controller import Robot

class EpuckDriver(CSVRobot):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        
        # Sensors
        self.ps = []
        for i in range(8):
            sensor_name = f'ps{i}'
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.time_step)
            self.ps.append(sensor)
            
        # Camera (Enabled for realism, though Supervisor handles logic)
        self.camera = self.getDevice('camera')
        if self.camera:
            self.camera.enable(self.time_step)
            
        # Motors
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def create_message(self):
        # Read sensors
        ps_values = [s.getValue() for s in self.ps]
        # Normalize sensors (0-4095 -> 0-1)
        # 4095 is close, 0 is far. We want 1 to be close (collision), 0 far.
        # Actually, let's just send raw values and let Supervisor/PPO handle normalization? 
        # Or normalize here. Standard e-puck ps range is roughly 0 to 4000+.
        # Let's normalize to [0, 1] where 1 is collision.
        normalized_ps = [val / 4096.0 for val in ps_values]
        
        return normalized_ps

    def use_message_data(self, message):
        # Message is [left_velocity, right_velocity]
        if message:
            try:
                left_speed = float(message[0])
                right_speed = float(message[1])
                
                # Clip speeds to max velocity (approx 6.28 rad/s for e-puck)
                max_speed = 6.28
                
                # Scale input [-1, 1] to [-max_speed, max_speed]
                # User requested faster movement.
                scale_factor = 6.0
                
                left_speed = max(min(left_speed * scale_factor, max_speed), -max_speed)
                right_speed = max(min(right_speed * scale_factor, max_speed), -max_speed)
                
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)
            except (ValueError, IndexError):
                pass

if __name__ == '__main__':
    robot_controller = EpuckDriver()
    robot_controller.run()
