"""
Part A - Base Code for Volleyball Serve Trajectory Analysis
=============================================================
Solves the equations of motion of a ball under gravity and air resistance,
using the Euler method, and computes:
    - Total flight time
    - Horizontal range where the ball lands
    - Whether the ball clears the net height at the net's x-position

The drag force is defined as: f = -(b*v + c*v^2) * v_hat
Since v_hat = v_vec / v, this can be rewritten as: f_vec = -(b + c*v) * v_vec
"""

import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (for saving files without a display)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical parameters and initial conditions
# ---------------------------------------------------------------------------
g = 9.81      # gravitational acceleration [m/s^2]
m = 0.27      # mass of a standard volleyball [kg]
b = 0.02      # linear drag coefficient [kg/s]
c = 0.02      # quadratic drag coefficient [kg/m]

# --- Court constants (per official volleyball rules) ---
NET_HEIGHT = 2.43   # net height for men's play [m] (women's: 2.24)
NET_X = 9.0          # distance from the serving point to the net [m] (approx. mid-court)
COURT_LENGTH = 18.0  # full court length [m]

dt = 0.001    # integration time step [s]


def simulate_serve(v0: float, alpha_deg: float, h: float,
                    g=g, m=m, b=b, c=c, dt=dt, t_max=10.0):
    """
    Simulates the ball's trajectory using the Euler method.

    Parameters:
        v0        - initial speed [m/s]
        alpha_deg - launch angle relative to the horizontal [degrees]
        h         - initial launch height [m]

    Returns a dict with: time array, position arrays, flight time, and horizontal range.
    """
    alpha = math.radians(alpha_deg)

    # initial conditions
    t = 0.0
    x = 0.0
    y = h
    vx = v0 * math.cos(alpha)
    vy = v0 * math.sin(alpha)

    ts, xs, ys = [t], [x], [y]

    while y > 0.0 and t < t_max:
        v = math.sqrt(vx**2 + vy**2)          # instantaneous speed magnitude
        drag_coef = (b + c * v) / m           # (b + c*v)/m -> shared between both axes

        ax = -drag_coef * vx
        ay = -g - drag_coef * vy

        # Euler method: update velocity first, then position
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        t = t + dt

        ts.append(t)
        xs.append(x)
        ys.append(y)

    # linear interpolation for the exact ground-impact moment (y=0)
    if len(ys) >= 2 and ys[-1] <= 0.0 < ys[-2]:
        y1, y2 = ys[-2], ys[-1]
        frac = y1 / (y1 - y2)
        t_land = ts[-2] + frac * (ts[-1] - ts[-2])
        x_land = xs[-2] + frac * (xs[-1] - xs[-2])
        ts[-1], xs[-1], ys[-1] = t_land, x_land, 0.0

    flight_time = ts[-1]
    horizontal_range = xs[-1]

    return {
        "t": ts, "x": xs, "y": ys,
        "flight_time": flight_time,
        "horizontal_range": horizontal_range,
    }


def height_at_x(result: dict, x_target: float):
    """Finds (via interpolation) the ball's height y at x=x_target along the trajectory."""
    xs, ys = result["x"], result["y"]
    for i in range(1, len(xs)):
        if xs[i - 1] <= x_target <= xs[i]:
            x1, x2 = xs[i - 1], xs[i]
            y1, y2 = ys[i - 1], ys[i]
            if x2 == x1:
                return y1
            frac = (x_target - x1) / (x2 - x1)
            return y1 + frac * (y2 - y1)
    return None  # the ball never reached this distance


def evaluate_serve(result: dict):
    """Checks the court constraints: clearing the net height and landing inside the court."""
    y_at_net = height_at_x(result, NET_X)

    passes_net = (y_at_net is not None) and (y_at_net > NET_HEIGHT)
    lands_in_court = result["horizontal_range"] <= COURT_LENGTH

    return {
        "height_at_net": y_at_net,
        "passes_net": passes_net,
        "lands_in_court": lands_in_court,
        "is_valid_serve": passes_net and lands_in_court,
    }


def find_valid_angle_range(v0: float, h: float, angle_min=0.5, angle_max=89.5, step=0.5):
    """
    Sweeps the launch angle (for fixed v0 and h) and finds the range of angles
    that produce a valid serve (clears the net AND lands inside the court).

    This is the basis of the feedback mechanism: it tells the player whether
    their actual angle was too low (ball hits the net) or too high
    (ball lands outside the court / too short).

    Returns a dict with:
        valid_angles      - sorted list of angles (deg) that produced a valid serve
        min_valid_angle   - smallest angle that clears the net (None if none found)
        max_valid_angle   - largest angle that still lands inside the court (None if none found)
    """
    valid_angles = []
    angle = angle_min
    while angle <= angle_max:
        result = simulate_serve(v0=v0, alpha_deg=angle, h=h)
        ev = evaluate_serve(result)
        if ev["is_valid_serve"]:
            valid_angles.append(angle)
        angle += step

    return {
        "valid_angles": valid_angles,
        "min_valid_angle": valid_angles[0] if valid_angles else None,
        "max_valid_angle": valid_angles[-1] if valid_angles else None,
    }


def build_feedback_message(v0: float, alpha_deg: float, h: float, evaluation: dict):
    """
    Builds a Hebrew feedback message for the player, based on:
      - whether the serve was valid (cleared the net + landed in court)
      - the actual angle vs. the range of angles that would have been valid
        for the same v0 and h

    Returns a tuple: (message_text, is_valid, angle_range_dict)
    """
    angle_range = find_valid_angle_range(v0=v0, h=h)
    min_a, max_a = angle_range["min_valid_angle"], angle_range["max_valid_angle"]

    if evaluation["is_valid_serve"]:
        message = (
            f"הגשה תקינה! הכדור עבר את הרשת ונחת בתוך המגרש. "
            f"בהינתן v0={v0:.1f} מ'/ש' וגובה h={h:.2f} מ', "
            f"טווח הזוויות התקינות הוא בין {min_a:.1f}° ל-{max_a:.1f}° "
            f"(את הגשת ב-{alpha_deg:.1f}°)."
        )
        return message, True, angle_range

    if not evaluation["passes_net"]:
        if min_a is not None:
            message = (
                f"הזווית שלך ({alpha_deg:.1f}°) נמוכה מדי - הכדור לא עובר את הרשת. "
                f"עבור המהירות והגובה הנוכחיים, כדאי להעלות את זווית ההגשה "
                f"לפחות ל-{min_a:.1f}° כדי לעבור את הרשת בבטחה."
            )
        else:
            message = (
                f"הזווית שלך ({alpha_deg:.1f}°) לא מספיקה כדי לעבור את הרשת, "
                f"וגם עם זוויות אחרות המהירות v0={v0:.1f} מ'/ש' לא מספיקה. "
                f"נסה להגביר את מהירות ההגשה."
            )
        return message, False, angle_range

    # passes the net but lands outside the court
    if max_a is not None:
        message = (
            f"הזווית שלך ({alpha_deg:.1f}°) גבוהה מדי - הכדור עובר את הרשת "
            f"אך נוחת מחוץ לגבולות המגרש. כדאי להנמיך את זווית ההגשה "
            f"עד ל-{max_a:.1f}° לכל היותר כדי שהכדור יישאר בתוך המגרש."
        )
    else:
        message = (
            f"הכדור עובר את הרשת אך נוחת מחוץ למגרש בזווית {alpha_deg:.1f}°. "
            f"נסה להפחית את מהירות ההגשה או להנמיך את הזווית."
        )
    return message, False, angle_range


def plot_trajectory(result: dict, title="Ball trajectory y(x)", filename="trajectory.png"):

    """Plots the ball's trajectory in the x-y plane, including the net marker."""
    plt.figure(figsize=(8, 5))
    plt.plot(result["x"], result["y"], linewidth=2, label="Ball trajectory")
    plt.axvline(NET_X, color="red", linestyle="--", alpha=0.6, label="Net position")
    plt.plot([NET_X], [NET_HEIGHT], "ro")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Horizontal distance x [m]")
    plt.ylabel("Height y [m]")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to file: {filename}")


# ---------------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: serve from height 1.80 m, speed 18 m/s, angle 10 degrees above horizontal
    v0 = 18.0
    alpha = 10.0
    h = 1.80

    result = simulate_serve(v0=v0, alpha_deg=alpha, h=h)
    evaluation = evaluate_serve(result)

    print("=" * 50)
    print(f"Initial conditions: v0={v0} m/s, alpha={alpha} deg, h={h} m")
    print("-" * 50)
    print(f"Total flight time:      {result['flight_time']:.3f} s")
    print(f"Horizontal range (landing): {result['horizontal_range']:.3f} m")
    print(f"Ball height above the net (x={NET_X} m): "
          f"{evaluation['height_at_net']:.3f} m (net height is {NET_HEIGHT} m)")
    print(f"Clears the net?      {'Yes' if evaluation['passes_net'] else 'No'}")
    print(f"Lands inside the court? {'Yes' if evaluation['lands_in_court'] else 'No'}")
    print(f"Valid serve?          {'Yes' if evaluation['is_valid_serve'] else 'No'}")
    print("=" * 50)

    plot_trajectory(result, title=f"Serve trajectory: v0={v0}, alpha={alpha} deg, h={h} m")
