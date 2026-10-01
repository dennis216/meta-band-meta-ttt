import pytest
from bfa.tusz_meta_ttt_v2.grouping import patient_order,nominal_lane_efficiency


@pytest.mark.parametrize('count',[1,3,4,15,463,579])
@pytest.mark.parametrize('bucket',[0,8,16,32])
def test_grouping_preserves_every_patient_and_partial_group_is_last(count,bucket):
    workloads={f'p{i}':(i+1)**2 for i in range(count)}
    result=patient_order(workloads,seed=3407,bucket_size=bucket)
    assert len(result)==count
    assert set(result)==set(workloads)
    assert result==patient_order(workloads,seed=3407,bucket_size=bucket)
    assert 0<nominal_lane_efficiency(result,workloads,2)<=1


def test_length_buckets_reduce_tail_waste_on_heterogeneous_workloads():
    work={f'p{i:03d}':2**(i//8) for i in range(80)}
    random_order=patient_order(work,seed=3407)
    bucket_order=patient_order(work,seed=3407,bucket_size=8)
    assert nominal_lane_efficiency(bucket_order,work,4)>nominal_lane_efficiency(random_order,work,4)
