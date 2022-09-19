import serial
import time
import threading
import math
import copy
from dataclasses import dataclass
from geopy import distance

from sleeptools import Rate


@dataclass
class Angle:
    _angle: float = 0
    _offset: float = 0

    def set(self, angle):
        angle += self._offset
        self._angle = self._zone(angle)

    def get(self):
        return self._angle

    def calibrate(self):
        self._offset = -self._angle

    def is_angle(self, angle, error=0):
        if isinstance(angle, Angle):
            angle = angle.get()
        angle = self._zone(angle)
        error = math.fabs(error)
        if error == 0:
            return angle == self._angle
        elif error > 180:
            error = 180

        a_low = self._angle - error
        a_high = self._angle + error
        low = self._zone(a_low)
        high = self._zone(a_high)

        is_low = False
        is_high = False
        if a_low < 0:
            is_low = low <= angle <= 360
        if a_high > 360:
            is_high = 0 <= angle <= high

        if is_low or is_high:
            return True
        elif a_low > 0 and a_high < 360:
            return low <= angle <= high
        return False

    def _zone(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle

@dataclass
class Position:
    x: float = 0
    y: float = 0
    latitude: float = 0
    longitude: float = 0
    heading: Angle = Angle()

    def coords(self):
        return (self.latitude, self.longitude)

    def distance_to(self, pos):
        return math.sqrt(pow(self.x - pos.x, 2) + pow(self.y - pos.y, 2))

    def geo_distance_to(self, pos):
        return distance.distance(self.coords(), pos.coords()).m

@dataclass
class GPSData:
    connected: bool = False
    new_packet: bool = False
    last_msg: float = 0

    second: int = 0
    minute: int = 0
    hour: int = 0
    day: int = 0
    month: int = 0
    year: int = 0
    latitude: int = 0
    longitude: int = 0
    groundSpeed: float = 0
    heading: float = 0
    status: str = ''
    mode: str = ''

@dataclass
class MotorData:
    connected: bool = False
    new_packet: bool = False
    last_msg: float = 0

    enc_left: int = 0
    enc_right: int = 0
    last_enc_left: int = 0
    last_enc_right: int = 0
    enc_ppr: int = 1176 * 25 / 12
    wheel_diameter: float = 0.2159
    wheel_circumference: float = wheel_diameter * math.pi
    enc_ppm: float = enc_ppr / wheel_circumference

    left_power: float = 0
    right_power: float = 0
    mow_power: float = 0

@dataclass
class IMUData:
    connected: bool = False
    new_packet: bool = False
    last_msg: float = 0

    pitch: Angle = Angle()
    roll: Angle = Angle()
    yaw: Angle = Angle()

    def calibrate(self):
        self.pitch.calibrate()
        self.roll.calibrate()
        self.yaw.calibrate()

@dataclass
class PositionLogger:
    angle_error: float = 5
    _position: Position = Position()
    _last_position: Position = Position()

    def set_pos(self, pos):
        self._position = pos
        if not pos.heading.is_angle(self._last_position.heading, error=self.angle_error):
            self._last_position = copy.copy(pos)

    def going_straight(self, dist=1):
        return dist <= self._position.distance_to(self._last_position)


class RobotController:
    START_CHAR = '$'
    GPS_MSG = "GPS"
    IMU_MSG = "IMU"
    MOTOR_MSG = "MOTORS"

    def __init__(self, connection_string, baudrate, timeout=0.5):
        self._connection_string = connection_string
        self._baudrate = baudrate
        self._timeout = timeout

        self._port = serial.Serial(connection_string, baudrate=baudrate,
                                   timeout=timeout, write_timeout=timeout)

        self._gps = GPSData()
        self._motors = MotorData()
        self._imu = IMUData()
        self.pos = Position()
        self._plog = PositionLogger()

        self._start = None
        self._startup = True
        self._startup_time = 5
        self._running = False

        self._update_thread = None
        self._lock = threading.Lock()

    def start(self):
        if not self._running:
            self._running = True
            self._start = time.perf_counter()
            self._startup = True
            self._update_thread = threading.Thread(target=self._update)
            self._update_thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._update_thread.join()

    def set_motor_speed(self, left_power=0, right_power=0, mow_power=0):
        if -1 <= left_power <= 1:
            self._motors.left_power = left_power
        if -1 <= right_power <= 1:
            self._motors.right_power = right_power
        if -1 <= mow_power <= 1:
            self._motors.mow_power = mow_power

    def reset_position(self):
        self.pos.x = 0
        self.pos.y = 0

    def _update(self):
        main_rate = Rate(150)
        motor_send_rate = Rate(30)
        pos_update_rate = Rate(30)
        last_pos = None
        while self._running:
            start = main_rate.get_start()
            if self._startup:
                if start - self._start >= self._startup_time:
                    self._startup = False
                    self._port.reset_input_buffer()
                    self._port.reset_output_buffer()
                else:
                    main_rate.sleep()
                    continue

            self._gps.connected = start - self._gps.last_msg < self._timeout
            self._motors.connected = start - self._motors.last_msg < self._timeout
            self._imu.connected = start - self._imu.last_msg < self._timeout

            while self._port.in_waiting > 0:
                str = self._port.readline()
                if len(str) == 0:
                    continue
                try:
                    str = str.decode('utf-8')
                    fields = str.split(',')
                    if fields[0][0] != self.START_CHAR:
                        continue
                except (UnicodeDecodeError, IndexError) as e:
                    pass

                try:
                    if fields[0][1:] == self.GPS_MSG:
                        self._gps.second = int(fields[1])
                        self._gps.minute = int(fields[2])
                        self._gps.hour = int(fields[3])
                        self._gps.day = int(fields[4])
                        self._gps.month = int(fields[5])
                        self._gps.year = int(fields[6])
                        if fields[8] != 'S':
                            self._gps.latitude = float(fields[7]) / 1e7
                        else:
                            self._gps.latitude = -float(fields[7]) / 1e7
                        if fields[10] != 'W':
                            self._gps.longitude = float(fields[9]) / 1e7
                        else:
                            self._gps.longitude = -float(fields[9]) / 1e7
                        self._gps.groundSpeed = int(fields[11]) / 1000
                        heading = int(fields[12]) / 1000
                        if heading == 0:
                            self._gps.heading = float('nan')
                        else:
                            self._gps.heading = heading
                        self._gps.status = fields[13]
                        self._gps.mode = fields[14]
                        self._gps.connected = True
                        self._gps.new_packet = True
                        self._gps.last_msg = start
                    elif fields[0][1:] == self.IMU_MSG:
                        self._imu.yaw.set(-int(fields[1]) / 1000)
                        self._imu.pitch.set(int(fields[2]) / 1000)
                        self._imu.roll.set(int(fields[3]) / 1000)
                        self._imu.connected = True
                        self._imu.new_packet = True
                        self._imu.last_msg = start
                    elif fields[0][1:] == self.MOTOR_MSG:
                        self._motors.enc_left = int(fields[1])
                        self._motors.enc_right = int(fields[2])
                        self._motors.connected = True
                        self._motors.new_packet = True
                        self._motors.last_msg = start
                except (ValueError, IndexError) as e:
                    pass

            if motor_send_rate.ready():
                lp = int(self._motors.left_power * 1000)
                rp = int(self._motors.right_power * 1000)
                mp = int(self._motors.mow_power * 1000)
                str = f"$JETSON,{lp},{rp},{mp},\n\r"
                try:
                    self._port.write(str.encode('utf-8'))
                except:
                    self.port.reset_output_buffer()

            if self._imu.new_packet:
                self._imu.new_packet = False

                dist = ((self._motors.enc_right - self._motors.last_enc_right) + (self._motors.enc_left - self._motors.last_enc_left)) / (2 * self._motors.enc_ppm)
                yaw = self._imu.yaw.get()

                self.pos.x += dist * math.cos(math.radians(yaw + 90))
                self.pos.y += dist * math.sin(math.radians(yaw + 90))
                self.pos.heading.set(yaw)

                self._motors.last_enc_left = self._motors.enc_left
                self._motors.last_enc_right = self._motors.enc_right

            if pos_update_rate.ready():
                self._plog.set_pos(self.pos)

            if self._gps.new_packet:
                self._gps.new_packet = False

                if last_pos is None:
                    last_pos = copy.copy(self.pos)

            main_rate.sleep()
        self._port.close()


if __name__ == "__main__":
    rc = RobotController("/dev/ttyACM0", 115200)
    rc.start()
    rate = Rate(100)
    try:
        while True:
            rate.sleep()
    except KeyboardInterrupt as e:
        pass
    rc.stop()
