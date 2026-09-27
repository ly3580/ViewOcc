"""Sample aligned source and target pairs for consistency training."""

import math

import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class ViewOccPairSampler(Sampler):
    """Emit ``source A/B, target A/B`` batches aligned by pair id."""

    def __init__(self, dataset, samples_per_gpu=4, num_replicas=None,
                 rank=None, seed=0, **kwargs):
        current_rank, world_size = get_dist_info()
        self.rank = current_rank if rank is None else rank
        self.num_replicas = world_size if num_replicas is None else num_replicas
        self.seed = 0 if seed is None else seed
        self.epoch = 0
        if samples_per_gpu != 4:
            raise ValueError('ViewOccPairSampler requires samples_per_gpu=4.')
        children = getattr(dataset, 'datasets', None)
        if children is None or len(children) != 2:
            raise ValueError('Pair training requires two concatenated datasets.')

        source_pairs = self._collect_pairs(children[0], offset=0)
        target_offset = len(children[0])
        target_pairs = self._collect_pairs(
            children[1], offset=target_offset, require_reference=True)
        self.batches = []
        for pair_id, target_pair in target_pairs:
            if pair_id not in source_pairs:
                raise ValueError(
                    f'Target reference_pair_id {pair_id} is missing.')
            self.batches.append(source_pairs[pair_id] + target_pair)
        if not self.batches:
            raise ValueError('No aligned source-target pairs were found.')
        self.num_batches = math.ceil(len(self.batches) / self.num_replicas)
        self.num_samples = self.num_batches * samples_per_gpu

    @staticmethod
    def _collect_pairs(dataset, offset, require_reference=False):
        """Group anchor and positive indices."""
        grouped = {}
        for index, info in enumerate(dataset.data_infos):
            pair_id = int(info['pair_id'])
            role = str(info['pair_role'])
            grouped.setdefault(pair_id, {})[role] = offset + index
        pairs = {}
        ordered = []
        for pair_id in sorted(grouped):
            roles = grouped[pair_id]
            if set(roles) != {'anchor', 'positive'}:
                raise ValueError(f'Pair {pair_id} must contain anchor/positive.')
            pair = [roles['anchor'], roles['positive']]
            if require_reference:
                infos = [
                    dataset.data_infos[index - offset] for index in pair]
                references = {
                    int(info['reference_pair_id']) for info in infos}
                if len(references) != 1:
                    raise ValueError('A target pair must share one reference.')
                ordered.append((references.pop(), pair))
            else:
                pairs[pair_id] = pair
        return ordered if require_reference else pairs

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            len(self.batches), generator=generator).tolist()
        total = self.num_batches * self.num_replicas
        order = (order * math.ceil(total / len(order)))[:total]
        start = self.rank * self.num_batches
        batches = [self.batches[index] for index in order[
            start:start + self.num_batches]]
        return iter([index for batch in batches for index in batch])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
