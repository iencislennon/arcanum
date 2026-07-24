"""
Phase 0 — финальная визуализация полного пайплайна:
симуляция -> GCC-PHAT/трилатерация -> фильтр Калмана.

Показывает на одном плане комнаты:
- положения микрофонов
- истинную (ground truth) траекторию движения источника
- сырые, шумные оценки позиции от TDOA-локализации
- сглаженную фильтром Калмана траекторию

Это визуальная проверка того, что вся логика Phase 0 работает согласованно,
прежде чем переходить к реальному железу (Phase 1).
"""

import numpy as np
import matplotlib.pyplot as plt

from signal_generator import (
    mic_array_positions_default,
    generate_trajectory,
    simulate_multichannel_recording,
)
from localization import localize_event
from kalman import ConstantVelocityKalman2D


def run_full_pipeline(waypoints, speed_m_s=1.0, event_interval=0.5, noise_level=0.2, fs=16000, seed=42):
    """Прогоняет весь пайплайн Phase 0 и возвращает все промежуточные данные для анализа/визуализации."""
    mics = mic_array_positions_default()
    traj_pos, traj_t = generate_trajectory(waypoints, speed_m_s=speed_m_s)

    recordings, events = simulate_multichannel_recording(
        mics, traj_pos, traj_t, fs=fs, event_interval_sec=event_interval,
        noise_level=noise_level, seed=seed,
    )

    chirp_duration = 0.05
    window_samples = int(chirp_duration * fs) + int(0.04 * fs)

    raw_positions, true_positions, timestamps = [], [], []
    for event in events:
        t_event = event["t"]
        true_positions.append(event["true_position"])
        timestamps.append(t_event)

        start_sample = int(t_event * fs)
        end_sample = start_sample + window_samples
        frame_per_mic = [recordings[m, start_sample:end_sample] for m in range(len(mics))]

        est_pos, _ = localize_event(frame_per_mic, fs, mics)
        raw_positions.append(est_pos)

    raw_positions = np.array(raw_positions)
    true_positions = np.array(true_positions)

    kf = ConstantVelocityKalman2D(initial_position=raw_positions[0], dt=event_interval)
    smoothed_positions = [raw_positions[0]]
    for pos in raw_positions[1:]:
        smoothed_positions.append(kf.step(pos))
    smoothed_positions = np.array(smoothed_positions)

    return {
        "mics": mics,
        "true_trajectory": traj_pos,
        "raw_positions": raw_positions,
        "smoothed_positions": smoothed_positions,
        "true_positions_at_events": true_positions,
        "timestamps": timestamps,
    }


def plot_static_summary(data, output_path):
    """Статичный график (не анимация) — весь путь целиком, для быстрой проверки."""
    fig, ax = plt.subplots(figsize=(9, 7.5))

    mics = data["mics"]
    ax.scatter(mics[:, 0], mics[:, 1], c="black", marker="^", s=120,
               label="Микрофоны", zorder=5)
    for i, m in enumerate(mics):
        ax.annotate(f"M{i}", (m[0], m[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)

    # истинная (плавная, полная) траектория движения
    true_traj = data["true_trajectory"]
    ax.plot(true_traj[:, 0], true_traj[:, 1], "g-", alpha=0.4, linewidth=2,
             label="Истинная траектория (ground truth)", zorder=1)

    # сырые TDOA-оценки в моменты событий
    raw = data["raw_positions"]
    ax.scatter(raw[:, 0], raw[:, 1], c="orange", marker="x", s=60,
               label="Сырые оценки TDOA", zorder=3, alpha=0.8)

    # сглаженная Калманом траектория
    smoothed = data["smoothed_positions"]
    ax.plot(smoothed[:, 0], smoothed[:, 1], "b-", linewidth=1.5, marker="o",
             markersize=4, label="Сглажено фильтром Калмана", zorder=4)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("X, метры")
    ax.set_ylabel("Y, метры")
    ax.set_title("Phase 0: симуляция TDOA-локализации + фильтр Калмана\n"
                  "(план комнаты, вид сверху)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Сохранено: {output_path}")

    # метрики точности
    true_at_events = data["true_positions_at_events"]
    raw_err = np.linalg.norm(raw - true_at_events, axis=1)
    smoothed_err = np.linalg.norm(smoothed - true_at_events, axis=1)

    print(f"\nМетрики точности по {len(true_at_events)} событиям:")
    print(f"  Сырые TDOA:  средняя ошибка = {np.mean(raw_err):.3f} м, макс = {np.max(raw_err):.3f} м")
    print(f"  Калман:      средняя ошибка = {np.mean(smoothed_err):.3f} м, макс = {np.max(smoothed_err):.3f} м")


if __name__ == "__main__":
    # Маршрут, похожий на реалистичное перемещение по комнате:
    # от двери (условно угол) к столу, потом к окну, без резких прямых углов
    waypoints = [
        (0.5, 0.5),   # "дверь"
        (2.0, 1.5),
        (2.5, 3.0),   # "стол"
        (3.5, 3.2),
        (4.5, 2.0),   # "окно"
    ]

    data = run_full_pipeline(waypoints, speed_m_s=0.8, event_interval=0.4, noise_level=0.15)
    plot_static_summary(data, "/home/claude/acoustic_sim/phase0_result.png")