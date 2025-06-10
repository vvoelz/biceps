

#ifndef TQDM_H
#define TQDM_H

#include <unistd.h>
#include <chrono>
#include <ctime>
#include <numeric>
#include <ios>
#include <string>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <atomic>

class tqdm {
private:
    std::mutex mtx;
    std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
    std::atomic<int> n_old{0};
    std::vector<double> deq_t;
    std::vector<int> deq_n;
    std::atomic<int> nupdates{0};
    std::atomic<int> total_{0};
    int period = 1;
    double period_lower_limit = 100.0;
    double target_rate = 25.0; // in Hz
    std::vector<const char*> bars = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
    bool is_tty = isatty(1);
    bool use_colors = true;
    bool color_transition = true;
    int width = 30;
    std::string right_pad = "▏";
    std::string label = "";
    bool in_jupyter = (getenv("JPY_PARENT_PID") != nullptr);

    std::atomic<int> current_task{0}; // Track the current task being displayed in Jupyter
    std::vector<int> tasks_completed; // Track completed tasks by task_id

    void hsv_to_rgb(float h, float s, float v, int& r, int& g, int& b) {
        if (s < 1e-6) {
            v *= 255.;
            r = v; g = v; b = v;
        }
        int i = (int)(h * 6.0);
        float f = (h * 6.) - i;
        int p = (int)(255.0 * (v * (1. - s)));
        int q = (int)(255.0 * (v * (1. - s * f)));
        int t = (int)(255.0 * (v * (1. - s * (1. - f))));
        v *= 255;
        i %= 6;
        int vi = (int)v;
        if (i == 0) { r = vi; g = t;  b = p; }
        else if (i == 1) { r = q;  g = vi; b = p; }
        else if (i == 2) { r = p;  g = vi; b = t; }
        else if (i == 3) { r = p;  g = q;  b = vi; }
        else if (i == 4) { r = t;  g = p;  b = vi; }
        else if (i == 5) { r = vi; g = p;  b = q; }
    }

public:
    tqdm(int num_tasks) : tasks_completed(num_tasks, 0) {
        if (in_jupyter) {
            period_lower_limit = 500.0;
            disable_colors();
        } else {
            color_transition = true;
            use_colors = true;
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        t_first = std::chrono::system_clock::now();
        t_old = std::chrono::system_clock::now();
        n_old = 0;
        deq_t.clear();
        deq_n.clear();
        nupdates = 0;
        total_ = 0;
        label = "";
        current_task = 0;
        std::fill(tasks_completed.begin(), tasks_completed.end(), 0); // Reset all tasks
    }

    void set_theme_line() { bars = {"─", "─", "─", "╾", "╾", "╾", "╾", "━", "═"}; }
    void set_theme_circle() { bars = {" ", "◓", "◑", "◒", "◐", "◓", "◑", "◒", "#"}; }
    void set_theme_braille() { bars = {" ", "⡀", "⡄", "⡆", "⡇", "⡏", "⡟", "⡿", "⣿" }; }
    void set_theme_braille_spin() { bars = {" ", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠇", "⠿" }; }
    void set_theme_vertical() { bars = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█"}; }
    void set_theme_basic() { bars = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"}; }
    void set_label(std::string label_) { label = label_; }
    void disable_colors() {
        color_transition = false;
        use_colors = false;
    }

    void finish(int task_id) {
        progress(task_id, total_, total_);
        tasks_completed[task_id] = 1; // Mark this task as completed
        current_task++; // Move to the next task in Jupyter
    }
    void close() { std::cout << std::endl; }

    void progress(int task_id, int curr, int tot) {
        std::lock_guard<std::mutex> lock(mtx); // Synchronize access to the progress display

        // In Jupyter, only show progress for the current task
        if (in_jupyter) {
            if (task_id != current_task || tasks_completed[task_id]) {
                return; // Skip this task if it's not the current one or already completed
            }
        }

        if ((curr % period == 0 || curr == tot) && tot > 0) {
            total_ = tot;
            nupdates++;
            auto now = std::chrono::system_clock::now();
            double dt_tot = std::chrono::duration<double>(now - t_first).count();
            double pct = (double)curr / tot * 100.0;
            double avg_rate = 0;
            double peta = 0;

            if (dt_tot > 0) {
                avg_rate = curr / dt_tot;
                peta = (tot - curr) / avg_rate;
                if (peta < 0) { peta = 0; }
            }

            double fills = (pct / 100.0) * width;
            int ifills = static_cast<int>(fills);

            if (nupdates > 2) {
                period = std::max(1, (int)(std::min(std::max((1.0 / target_rate) * curr / dt_tot, 1.0), period_lower_limit)));
            }

            if ((tot - curr) <= period || curr == tot) {
                pct = 100.0;
                avg_rate = tot / dt_tot;
                peta = 0;
            }

            //printf("\r\015 ");
            printf("\015 ");
            if (use_colors && !in_jupyter) {
                if (color_transition) {
                    int r = 255, g = 255, b = 255;
                    hsv_to_rgb(0.0 + 0.01 * pct / 3, 0.65, 1.0, r, g, b);
                    printf("\033[38;2;%d;%d;%dm ", r, g, b);
                } else {
                    printf("\033[32m ");
                }
            }

            for (int i = 0; i < ifills; i++) std::cout << bars[8];
            if ((curr != tot)) printf("%s", bars[(int)(8.0 * (fills - ifills))]);
            for (int i = 0; i < width - ifills - 1; i++) std::cout << bars[0];
            printf("%s ", right_pad.c_str());
            if (use_colors && !in_jupyter) { printf("\033[1m\033[31m"); }
            printf("%4.1f%% ", pct);
            if (use_colors and !in_jupyter) { printf("\033[34m"); }

            std::string unit = "Hz";
            double div = 1.;
            if (avg_rate > 1e6) { unit = "MHz"; div = 1.0e6; }
            else if (avg_rate > 1e3) { unit = "kHz"; div = 1.0e3; }
            printf("[%4d/%4d | %3.1f %s | %d | %.0fs | %.0fs] ", curr, tot, avg_rate / div, unit.c_str(), task_id+1, peta, dt_tot);
            printf("%s ", label.c_str());
            if (use_colors && !in_jupyter) printf("\033[0m\033[32m\033[0m\015 ");

            std::cout.flush(); // Ensure output is flushed to the terminal

            // If task is complete in Jupyter, move to the next task
            if (in_jupyter && curr == tot) {
                tasks_completed[task_id] = 1; // Mark this task as completed
                current_task++; // Move to the next task
            }
        }
    }
};

#endif


