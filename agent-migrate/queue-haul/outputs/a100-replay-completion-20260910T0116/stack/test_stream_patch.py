import asyncio
import pytest


def test_native_delta_events_remain_distinct():
    from vllm.v1.engine.output_processor import RequestOutputCollector
    from vllm.sampling_params import RequestOutputKind
    from vllm.outputs import RequestOutput,CompletionOutput
    def output(token):
        return RequestOutput('r',None,[1],None,[CompletionOutput(0,'x',[token],None,None)],False)
    async def check():
        q=RequestOutputCollector(RequestOutputKind.DELTA,'r')
        q.put(output(7));q.put(output(8))
        assert (await q.get()).outputs[0].token_ids==[7]
        assert (await q.get()).outputs[0].token_ids==[8]
        assert q.get_nowait() is None
        q.put(ValueError('retained error'))
        with pytest.raises(ValueError,match='retained error'):await q.get()
    asyncio.run(check())
