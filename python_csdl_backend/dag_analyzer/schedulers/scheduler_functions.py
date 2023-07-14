def add_to_schedule(
        full_schedule: list[list[str]],
        node: str,
        rank: int
    ):

    full_schedule[rank].append(node)

def add_to_estimated_timeline(
        schedule_estimated_timeline: list[list[str]],
        estimated_time,
        rank: int,
    ):

    schedule_estimated_timeline[rank].append(estimated_time)

def schedule_pt2pt_comm(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        node: str,
        communication_cost,
        from_rank: int,
        to_rank:int,
        pt2pts:dict = None,
    ):
    """
    Schedule a point-to-point communication of a variable between ranks to full_schedule.
    Update rank times
    """


    # Find what point on both ranks this communication occurs
    # We assume no buffer !!! 
    # Therefore, the communication send and recieve occurs at the same time on both processors when they are avaialble. 

    # Update
    if schedule_estimated_timeline[from_rank][-1] >= schedule_estimated_timeline[to_rank][-1]:
        waiting_rank = to_rank
        bottleneck_rank = from_rank
        sync_time = schedule_estimated_timeline[from_rank][-1]
    else:
        waiting_rank = from_rank
        bottleneck_rank = to_rank
        sync_time = schedule_estimated_timeline[to_rank][-1]

    # Add waiting from one rank
    add_to_schedule(full_schedule, f'WAITING_for_{bottleneck_rank}_for_{node}', waiting_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, sync_time, waiting_rank)

    # Now add the pt to pt communication operation
    add_to_schedule(full_schedule, f'SEND_/{node}/_to_/{to_rank}/_from_/{from_rank}', from_rank)
    add_to_schedule(full_schedule, f'GET_/{node}/_from_/{from_rank}/_to_/{to_rank}', to_rank)
    comm_time = sync_time + communication_cost
    add_to_estimated_timeline(schedule_estimated_timeline, comm_time, from_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, comm_time, to_rank)

    if pt2pts is not None:
        # keep note that we scheduled
        pt2pts[get_pt2pts_key(from_rank, to_rank)]['scheduled'].add(node)


def schedule_pt2pt_comm_nb(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        node: str,
        communication_cost,
        from_rank: int,
        to_rank:int,
        pt2pts:dict = None,
    ):
    """
    Schedule a point-to-point communication of a variable between ranks to full_schedule.
    Update rank times
    """
    # Find what point on both ranks this communication occurs
    # We assume no buffer !!! 
    # Therefore, the communication send and recieve occurs at the same time on both processors when they are avaialble. 
    
    # Update
    if schedule_estimated_timeline[from_rank][-1] >= schedule_estimated_timeline[to_rank][-1]:
        waiting_rank = to_rank
        bottleneck_rank = from_rank
        sync_time = schedule_estimated_timeline[from_rank][-1]
    else:
        waiting_rank = from_rank
        bottleneck_rank = to_rank
        sync_time = schedule_estimated_timeline[to_rank][-1]

    # Add waiting from one rank
    add_to_schedule(full_schedule, f'WAITING_for_{bottleneck_rank}_for_{node}', waiting_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, sync_time, waiting_rank)

    # Now add the pt to pt communication operation
    add_to_schedule(full_schedule, f'SEND_/{node}/_to_/{to_rank}/', from_rank)
    add_to_schedule(full_schedule, f'GET_/{node}/_from_/{from_rank}/', to_rank)
    comm_time = sync_time + communication_cost
    add_to_estimated_timeline(schedule_estimated_timeline, comm_time, from_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, comm_time, to_rank)

    if pt2pts is not None:
        # keep note that we scheduled
        pt2pts[get_pt2pts_key(from_rank, to_rank)]['scheduled'].add(node)


def schedule_operation(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        operation: str,
        operation_cost: float,
        rank: int,
    ):
    """
    Schedule an operation to rank's schedule.
    """
    if operation_cost < 0:
        raise ValueError(f'operation cost ({operation}) cannot be negative')
    add_to_schedule(full_schedule, operation, rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[rank][-1]+operation_cost, rank)

def get_pt2pts_key(rank1, rank2):
    if rank1 < rank2:
        return (rank1, rank2)
    elif rank2 < rank1:
        return (rank2, rank1)
    else:
        raise ValueError('sdjs')

def schedule_pre_irecieve(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        recieve_cost: float,
        to_rank: int,
        from_rank:int,
    ):

    add_to_schedule(full_schedule, f'IRECV_/{variable}/_from_/{from_rank}/', to_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[to_rank][-1]+recieve_cost, to_rank)

def schedule_send_only(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        send_cost: float,
        to_rank: int,
        from_rank:int,
    ):

    add_to_schedule(full_schedule, f'SsENDONLY_/{variable}/_to_/{to_rank}/', from_rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[from_rank][-1]+send_cost, from_rank)


def schedule_irecv_wait(
        full_schedule: list[list[str]],
        schedule_estimated_timeline: list[list[int]],
        variable: str,
        rank = int,
    ):

    add_to_schedule(full_schedule, f'irecvwait_/{variable}', rank)
    add_to_estimated_timeline(schedule_estimated_timeline, schedule_estimated_timeline[rank][-1], rank)