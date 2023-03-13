# Many-Producers-to-Many-Consumers

## Known issue:
```
FakeExperienceBuffer(
            cpu_offload=False,
            )
```
`cpu_offload=False` raise error.

## How to run
```
bash run.sh
```
=
```
python main.py --world_size 5 --maker_time 3.0 --trainer_time 2.0 --port 29604
```

- world_size: number of workers. in main.py, rank_to_role define the role of each rank, from 0 to world_size.
- maker_time: sec for one maker (producer) to yield an exp
- trainer_time: sec for one trainer (consumer) to eat an exp.