"""Label-independent patient grouping to reduce unequal lane runtimes."""
import random


def patient_order(workloads, *, seed, bucket_size=0):
    rng=random.Random(seed)
    if not bucket_size:
        result=sorted(workloads)
        rng.shuffle(result)
        return result
    if bucket_size<4 or bucket_size%4:
        raise ValueError('Patient bucket size must be a positive multiple of four')
    ordered=sorted(workloads,key=lambda p:(workloads[p],p))
    groups=[]
    remainder=[]
    for start in range(0,len(ordered),bucket_size):
        bucket=ordered[start:start+bucket_size]
        rng.shuffle(bucket)
        for i in range(0,len(bucket),4):
            group=bucket[i:i+4]
            if len(group)==4: groups.append(group)
            else: remainder.extend(group)
    rng.shuffle(groups)
    # Keep the only partial outer group at the end; flattening an early partial
    # group would accidentally mix all later four-patient optimizer boundaries.
    return [p for group in groups for p in group]+remainder


def nominal_lane_efficiency(order,workloads,patients_per_batch):
    capacity=0
    useful=0
    for outer in range(0,len(order),4):
        group=order[outer:outer+4]
        for start in range(0,len(group),patients_per_batch):
            lanes=group[start:start+patients_per_batch]
            useful+=sum(workloads[p] for p in lanes)
            capacity+=len(lanes)*max(workloads[p] for p in lanes)
    return useful/capacity if capacity else 1.
