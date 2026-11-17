## Uruchomienie

Pierwszy terminal
```bash
ros2 launch arm05_sim world.launch.py
```

Drugi terminal
```bash
ros2 launch nav2_bringup rviz_launch.py use_sim_time:=true
```

Trzeci terminal
```bash
ros2 launch arm05_explorer explorer.launch.py
```

Czwarty terminal
```bash
ros2 action send_goal /find_cubes arm05_interfaces/action/FindCubes "{cubes_to_find: 5, timeout_sec: 900.0}" --feedback
```

Sprawdzenie wyników z rzeczywistym pomiarem
```bash
ros2 run arm05_explorer evaluate
```
