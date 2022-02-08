# %%
import numpy as np


class Car:
    def __init__(
        self,
        init_pos,
        init_lane,
        init_speed,
        desired_speed,
    ):
        self.length = 5
        self.natural_acceleration = 5
        self.natural_deceleration = 8
        self.delta = 1.5  # coefficient de variation de l'acceleration avec la vitesse
        self.is_first = False
        self.s0 = 30  # distance minimale de sécurité désirée
        self.desired_speed = desired_speed  # km.h
        self.init_pos = init_pos
        self.init_speed = init_speed
        self.init_lane = init_lane
        # self.init_acc = init_acc
        self.lane = self.init_lane
        self.T = 1.5  # Temps minimal de parcours entre deux voitures
        self.anti_oscillation_count = 0

        self.pos = self.init_pos
        self.speed = self.init_speed
        # self.acceleration = self.init_acc

    def varvitesse(self, front_car):
        if self.is_first:
            dv = self.natural_acceleration * (
                1 - (self.speed / self.desired_speed) ** self.delta
            )
        else:
            ecart = max(abs(front_car.pos - self.pos), 25)

            st = (
                self.s0
                + self.speed * self.T
                - (self.speed * (front_car.speed - self.speed))
                / (2 * np.sqrt(self.natural_acceleration * self.natural_deceleration))
            )

            dv = self.natural_acceleration * (
                1 - (self.speed / self.desired_speed) ** self.delta - (st / ecart) ** 2
            )
        return dv

    def doubler(
        self,
        front_car,
        front_car_left_lane,
        rear_car_left_lane,
        front_car_right_lane,
        rear_car_right_lane,
        lane_nb,
        dt,
    ):
        ecart = abs(self.pos - front_car.pos)
        ecart1dr = abs(self.pos - rear_car_left_lane.pos)
        ecart1dv = abs(self.pos - front_car_left_lane.pos)
        ecart0dr = abs(self.pos - rear_car_right_lane.pos)
        ecart0dv = abs(self.pos - front_car_right_lane.pos)
        res = 0
        if self.anti_oscillation_count != 0:
            self.anti_oscillation_count -= 1
        if (
            self.lane + 1 < lane_nb
            and front_car.speed < self.desired_speed
            and ecart < 8 * self.speed
            and front_car is not self
            and self.anti_oscillation_count == 0
        ):
            if (
                ecart1dr > rear_car_left_lane.speed * 2 * self.T
                or rear_car_left_lane is self
            ):
                if (
                    ecart1dv > front_car_left_lane.speed * 2 * self.T
                    or front_car_left_lane is self
                ):
                    res = +1
                    self.anti_oscillation_count = int(15 / dt)
        elif self.lane > 0 and self.anti_oscillation_count == 0:
            if (
                ecart0dr > rear_car_right_lane * self.T or rear_car_right_lane is self
            ):  # distance correspondant à 2 sec a la vitesse
                if (
                    ecart0dv > front_car_right_lane.speed * self.T
                    and front_car_right_lane.speed + 1 >= self.speed
                ) or front_car_right_lane is self:
                    res = -1
                    self.anti_oscillation_count = int(15 / dt)

        return res

    def veh_alentours(self, cars):
        indvdv = self
        indvdr1 = self
        indvdv1 = self
        indvdv0 = self
        indvdr0 = self
        for car in cars:
            if car.lane == self.lane and self.pos < car.pos:
                if car.pos <= indvdv.pos or indvdv is self:
                    indvdv = car
            if car.lane == self.lane + 1:
                if car.pos < self.pos:
                    if car.pos >= indvdr1.pos or indvdr1 is self:
                        indvdr1 = car
                if self.pos - car.pos < 0:
                    if car.pos <= indvdv1.pos or indvdv1 is self:
                        indvdv1 = car
            if car.lane == self.lane - 1:
                if car.pos < self.pos:
                    if car.pos >= indvdr0.pos or indvdr0 is self:
                        indvdr0 = car
                if self.pos - car.pos < 0:
                    if car.pos <= indvdv0.pos or indvdv0 is self:
                        indvdv0 = car

        return indvdv, indvdv1, indvdr1, indvdv0, indvdr0

    def step(self, xm, cars, car_to_delete, dt, nb_lane, first_car_in_lane):
        if self.pos > 0.95 * xm:
            car_to_delete.append(self)
            if first_car_in_lane[self.lane] is self:
                first_car_in_lane[self.lane] = None

        vdv, vdv1, vdr1, vdv0, vdr0 = self.veh_alentours(cars)
        double = self.doubler(vdv, vdr1, vdv1, vdv0, vdr0, nb_lane, dt)

        speed_saved = self.speed
        self.speed = abs(self.speed + dt * self.varvitesse(vdv))
        self.pos = self.pos + dt * (self.speed + speed_saved) / 2

        if double != 0 and first_car_in_lane[self.lane] is self:
            first_car_in_lane[self.lane] = vdv
        self.lane += double
        if (
            first_car_in_lane[self.lane] is None
            or self.pos < first_car_in_lane[self.lane].pos
        ):
            first_car_in_lane[self.lane] = self

        return None


# %%
import random


def attribueVitesse(d, v):
    a = []
    if d > 1:
        for i in range(d):
            c = random.randint(85, 105)
            a.append(c * v / 100)
        return a
    elif d == 1:
        c = random.randint(85, 105)
        return (c * v) / 100


def etat_initial(n_cars, car_length, initial_car_distance, speed_limit, lane_nb):
    init_positions = np.arange(
        0,
        -(n_cars - 1) * (car_length + initial_car_distance) - 1,
        -(car_length + initial_car_distance),
    )  # Positions initiales des voitures
    init_speeds = attribueVitesse(n_cars, speed_limit)
    init_desired_speeds = attribueVitesse(n_cars, speed_limit)
    cars = [
        Car(pos, 0, speed, desired_speed)
        for pos, speed, desired_speed in zip(
            init_positions, init_speeds, init_desired_speeds
        )
    ]
    first_car_in_lane = [None] * lane_nb
    first_car_in_lane[0] = cars[n_cars - 1]
    return cars, first_car_in_lane


# Simulation parameters


car_params = {
    "car_length": 5,  # longueur des voitures (m)
    "T": 1.5,  # temps minimal de parcours de l'intervalle entre deux voitures
    "aa": 5,  # acceleration naturelle
    "bb": 8,  # deceleration
    "delta": 1.5,  # coefficient de variation de l'acceleration avec la vitesse
    "s0": 30,  # distance minimale entre les voitures
}

n_cars = 3  # nombre de voitures au départ
initial_car_distance = 50  # distance initiale entre les voitures (m)
dt = 1 / 20  # time step
road_length = 1100
road_speed_limit = 130 / 3.6  # 130km.h in m.s
n_lane = 2


# %% SIMULATION
cars, first_car_in_lane = etat_initial(
    n_cars, car_params["car_length"], initial_car_distance, road_speed_limit, n_lane
)


def simul_step(cars):
    car_to_delete = []
    for car in cars:
        car.step(road_length, cars, car_to_delete, dt, n_lane, first_car_in_lane)

    for car in car_to_delete:
        cars.remove(car)


# for time in range(100):
#     print(time)
#     simul_step(cars)

# %% matplotlib ipympl

from matplotlib import animation
import matplotlib.pyplot as plt


def animate(i):
    simul_step(cars)
    graph = ax.scatter(
        [car.pos for car in cars], [car.lane for car in cars], marker="s", color="blue"
    )
    return graph


def init():
    graph = ax.scatter([], [], animated=True)
    return graph


fig = plt.figure()  # initialise la figure

fig.clf()

ax = fig.add_subplot(
    111, autoscale_on=False, xlim=(-road_length - 1, road_length), ylim=(-0.3, n_lane)
)


ani = animation.FuncAnimation(
    fig, animate, frames=500, interval=1, blit=False, init_func=init
)

plt.yticks(
    [i for i in range(n_lane)],
    ["Voie " + str(i + 1) for i in range(n_lane)],
    size="medium",
)
plt.show()
# %%
