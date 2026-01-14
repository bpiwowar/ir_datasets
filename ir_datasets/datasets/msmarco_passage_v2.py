import re
import os
import contextlib
import gzip
import io
from pathlib import Path
import json
from typing import NamedTuple, Tuple
import tarfile
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore, DEFAULT_DOCSTORE_OPTIONS, FileAccess
from ir_datasets.util import Cache, DownloadConfig, GzipExtract, Lazy, Migrator, TarExtractAll
from ir_datasets.datasets.base import Dataset, YamlDocumentation, FilteredQueries, FilteredScoredDocs, FilteredQrels
from ir_datasets.formats import TsvQueries, TrecQrels, TrecScoredDocs, BaseDocs
from ir_datasets.datasets.msmarco_passage import DUA, DL_HARD_QIDS_BYFOLD, DL_HARD_QIDS, TREC_DL_QRELS_DEFS

QRELS_DEFS = {
    1: 'Based on mapping from v1 of MS MARCO'
}

_logger = ir_datasets.log.easy()

NAME = 'msmarco-passage-v2'


class MsMarcoV2Passage(NamedTuple):
    doc_id: str
    text: str
    spans: Tuple[Tuple[int, int], ...]
    msmarco_document_id: str
    def default_text(self):
        """
        text
        """
        return self.text


def parse_msmarco_passage(line):
    data = json.loads(line)
    # extract spans in the format of "(123,456),(789,101123)"
    spans = tuple((int(a), int(b)) for a, b in re.findall(r'\((\d+),(\d+)\)', data['spans']))
    return MsMarcoV2Passage(
        data['pid'],
        data['passage'],
        spans,
        data['docid'])


class MsMarcoV2Passages(BaseDocs):
    def __init__(self, dlc, pos_dlc=None):
        super().__init__()
        self._dlc = dlc
        self._pos_dlc = pos_dlc

    @ir_datasets.util.use_docstore
    def docs_iter(self):
        if self._pos_dlc is not None:
            # the shortcut only applies if the default pos
            # files are used (i.e., no filtering is applied)
            yield from self.docs_store()
        else:
            with self._dlc.stream() as stream, \
                 tarfile.open(fileobj=stream, mode='r|') as tarf:
                for record in tarf:
                    if not record.name.endswith('.gz'):
                        continue
                    file = tarf.extractfile(record)
                    with gzip.open(file) as file:
                        for line in file:
                            yield parse_msmarco_passage(line)

    def docs_cls(self):
        return MsMarcoV2Passage

    def docs_store(self, field='doc_id', options=DEFAULT_DOCSTORE_OPTIONS):
        return PickleLz4FullStore(
            path=f'{self.docs_path()}.pklz4',
            init_iter_fn=self.docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            key_field_prefix='msmarco_passage_', # cut down on storage by removing prefix in lookup structure
            index_fields=['doc_id'],
            count_hint=138_354_198,
            options=options
        )

    def docs_count(self):
        return self.docs_store().count()

    def docs_namespace(self):
        return NAME

    def docs_lang(self):
        return 'en'

    def docs_path(self):
        return self._dlc.path()


def _init():
    base_path = ir_datasets.util.home_path()/NAME
    documentation = YamlDocumentation(f'docs/{NAME}.yaml')
    dlc = DownloadConfig.context(NAME, base_path, dua=DUA)
    subsets = {}
    collection = MsMarcoV2Passages(dlc['passages'])

    qrels_migrator = Migrator(base_path/'qrels_version.txt', 'v2',
        affected_files=[base_path/'train'/'qrels.tsv', base_path/'dev1'/'qrels.tsv', base_path/'dev2'/'qrels.tsv'],
        message='Updating qrels (task organizers removed duplicates)')

    subsets['dedup'] = Dataset(
        MsMarcoV2Passages(dlc['passages'], TarExtractAll(dlc['dedup_positions'], base_path/'dedup_positions'))
    )

    subsets['train'] = Dataset(
        collection,
        TsvQueries(dlc['train/queries'], namespace='msmarco', lang='en'),
        qrels_migrator(TrecQrels(dlc['train/qrels'], QRELS_DEFS)),
        TrecScoredDocs(GzipExtract(dlc['train/scoreddocs'])),
    )
    subsets['dev1'] = Dataset(
        collection,
        TsvQueries(dlc['dev1/queries'], namespace='msmarco', lang='en'),
        qrels_migrator(TrecQrels(dlc['dev1/qrels'], QRELS_DEFS)),
        TrecScoredDocs(GzipExtract(dlc['dev1/scoreddocs'])),
    )
    subsets['dev2'] = Dataset(
        collection,
        TsvQueries(dlc['dev2/queries'], namespace='msmarco', lang='en'),
        qrels_migrator(TrecQrels(dlc['dev2/qrels'], QRELS_DEFS)),
        TrecScoredDocs(GzipExtract(dlc['dev2/scoreddocs'])),
    )
    subsets['trec-dl-2021'] = Dataset(
        collection,
        TsvQueries(dlc['trec-dl-2021/queries'], namespace='msmarco', lang='en'),
        TrecQrels(dlc['trec-dl-2021/qrels'], TREC_DL_QRELS_DEFS),
        TrecScoredDocs(GzipExtract(dlc['trec-dl-2021/scoreddocs'])),
    )
    dl21_judged = Lazy(lambda: {q.query_id for q in subsets['trec-dl-2021'].qrels_iter()})
    subsets['trec-dl-2021/judged'] = Dataset(
        FilteredQueries(subsets['trec-dl-2021'].queries_handler(), dl21_judged),
        FilteredScoredDocs(subsets['trec-dl-2021'].scoreddocs_handler(), dl21_judged),
        subsets['trec-dl-2021'],
    )
    subsets['trec-dl-2022'] = Dataset(
        collection,
        TsvQueries(dlc['trec-dl-2022/queries'], namespace='msmarco', lang='en'),
        TrecQrels(dlc['trec-dl-2022/qrels'], TREC_DL_QRELS_DEFS),
        TrecScoredDocs(GzipExtract(dlc['trec-dl-2022/scoreddocs'])),
    )
    dl22_judged = Lazy(lambda: {q.query_id for q in subsets['trec-dl-2022'].qrels_iter()})
    subsets['trec-dl-2022/judged'] = Dataset(
        FilteredQueries(subsets['trec-dl-2022'].queries_handler(), dl22_judged),
        FilteredScoredDocs(subsets['trec-dl-2022'].scoreddocs_handler(), dl22_judged),
        subsets['trec-dl-2022'],
    )
    subsets['trec-dl-2023'] = Dataset(
        collection,
        TsvQueries(dlc['trec-dl-2023/queries'], namespace='msmarco', lang='en'),
        TrecScoredDocs(GzipExtract(dlc['trec-dl-2023/scoreddocs'])),
    )

    ir_datasets.registry.register(NAME, Dataset(collection, documentation("_")))
    for s in sorted(subsets):
        ir_datasets.registry.register(f'{NAME}/{s}', Dataset(subsets[s], documentation(s)))

    return collection, subsets


collection, subsets = _init()
