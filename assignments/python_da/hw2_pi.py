import random


def pi_approx_mc(no_steps, print_every=50):
    no_points_within_circle = 0

    for step_number in range(1, no_steps + 1):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x ** 2 + y ** 2 <= 1:
            no_points_within_circle += 1

        # Step number is equal to number of point within square ([-1, -1], [-1, 1], [1, -1], [1, 1]).
        pi = 4 * no_points_within_circle / step_number
        if (step_number % print_every) == 0:
            print(f"After {step_number} steps: pi={pi}")
    return pi


if __name__ == "__main__":
    print('Total number of steps:')
    no_steps_ = int(input())
    print('Period (in steps) after which print current value')
    print_every_ = int(input())
    print(f"Result: {pi_approx_mc(no_steps_, print_every_)}")
