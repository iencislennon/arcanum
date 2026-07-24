"""
Модуль локализации источника звука.

Два шага:
1. GCC-PHAT — для пары микрофонов оценивает задержку (TDOA) между приходом
   одного и того же звука на оба микрофона.
2. Трилатерация — по набору TDOA от нескольких пар микрофонов методом
   наименьших квадратов восстанавливает 2D координаты источника.
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import least_squares

SPEED_OF_SOUND = 343.0


def gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Generalized Cross-Correlation with Phase Transform.

    Возвращает оценку задержки tau (в секундах): на сколько сигнал sig2
    задержан относительно sig1. Положительный tau означает, что sig2
    пришёл позже sig1 (источник ближе к микрофону 1).

    PHAT-взвешивание (нормировка кросс-спектра по модулю) делает оценку
    устойчивой к реверберации и разнице в амплитудах — поэтому это
    стандартный выбор для TDOA в помещениях, а не просто обычная
    кросс-корреляция.
    """
    n = len(sig1) + len(sig2)
    X1 = fft(sig1, n=n)
    X2 = fft(sig2, n=n)

    R = X1 * np.conj(X2)
    denom = np.abs(R)
    denom[denom < 1e-10] = 1e-10  # защита от деления на ноль в тишине
    R /= denom

    cc = np.real(ifft(R))

    max_shift = int(fs * max_tau) if max_tau else n // 2
    max_shift = min(max_shift, n // 2)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(fs)
    return tau


def compute_all_tdoas(frame_per_mic, fs, mic_positions, max_room_diagonal=8.0):
    """
    Считает TDOA для всех пар микрофонов на одном "фрейме" (коротком окне
    сигнала, где предположительно есть одно звуковое событие).

    max_room_diagonal ограничивает разумный диапазон поиска задержки —
    звук физически не может задержаться больше, чем время прохождения
    диагонали помещения, это отсекает заведомо ложные пики корреляции.

    Возвращает список ((mic_i, mic_j), tau) для всех пар.
    """
    n_mics = len(frame_per_mic)
    max_tau = max_room_diagonal / SPEED_OF_SOUND

    tdoas = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            tau = gcc_phat(frame_per_mic[i], frame_per_mic[j], fs, max_tau=max_tau)
            tdoas.append(((i, j), tau))
    return tdoas


def estimate_position(mic_positions, tdoas, initial_guess=None):
    """
    Трилатерация методом наименьших квадратов.

    Для каждой пары микрофонов (i, j) с измеренным TDOA tau_ij,
    "предсказанный" TDOA при гипотетической позиции source — это
    (dist(source, mic_i) - dist(source, mic_j)) / speed_of_sound.

    Минимизируем сумму квадратов разницы между измеренным и предсказанным
    TDOA по всем парам -> находим наиболее вероятную позицию источника.
    """
    if initial_guess is None:
        # хорошая стартовая точка - центр микрофонного массива
        initial_guess = np.mean(mic_positions, axis=0)

    def residuals(pos):
        errs = []
        for (i, j), tau_measured in tdoas:
            d_i = np.linalg.norm(pos - mic_positions[i])
            d_j = np.linalg.norm(pos - mic_positions[j])
            tau_predicted = (d_i - d_j) / SPEED_OF_SOUND
            errs.append(tau_predicted - tau_measured)
        return errs

    result = least_squares(residuals, x0=initial_guess, method="lm")
    return result.x, result.cost  # позиция + остаточная невязка (мера уверенности)


def localize_event(frame_per_mic, fs, mic_positions):
    """
    Полный пайплайн для одного события: TDOA по всем парам -> трилатерация.
    Обёртка для удобства использования извне.
    """
    tdoas = compute_all_tdoas(frame_per_mic, fs, mic_positions)
    position, cost = estimate_position(mic_positions, tdoas)
    return position, cost


if __name__ == "__main__":
    from signal_generator import (
        mic_array_positions_default,
        generate_trajectory,
        simulate_multichannel_recording,
    )

    fs = 16000
    mics = mic_array_positions_default()
    waypoints = [(0.5, 0.5), (2.5, 3.5), (4.5, 0.5)]
    traj_pos, traj_t = generate_trajectory(waypoints, speed_m_s=1.0)
    recordings, events = simulate_multichannel_recording(mics, traj_pos, traj_t, fs=fs)

    print("Проверка локализации на нескольких синтетических событиях:\n")
    chirp_duration = 0.05
    window_samples = int(chirp_duration * fs) + int(0.02 * fs)  # чирп + запас на задержку

    errors = []
    for event in events[:8]:
        t_event = event["t"]
        true_pos = event["true_position"]

        # вырезаем окно вокруг события из каждого канала
        start_sample = int(t_event * fs)
        end_sample = start_sample + window_samples + int(0.02 * fs)  # запас на TDOA между мик.
        frame_per_mic = [recordings[m, start_sample:end_sample] for m in range(len(mics))]

        est_pos, cost = localize_event(frame_per_mic, fs, mics)
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)

        print(
            f"t={t_event:.2f}s | истина=({true_pos[0]:.2f},{true_pos[1]:.2f}) "
            f"| оценка=({est_pos[0]:.2f},{est_pos[1]:.2f}) | ошибка={error:.3f} м | cost={cost:.2e}"
        )

    print(f"\nСредняя ошибка локализации: {np.mean(errors):.3f} м")
    print(f"Макс. ошибка: {np.max(errors):.3f} м")