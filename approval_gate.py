import asyncio

from dataclasses import dataclass,field

@dataclass
class ApprovalGate:
    event:asyncio.Event = field(default_factory=asyncio.Event)
    rejected:bool = False

    def approval(self):
        self.rejected =False
        self.event.set()

    def reject(self):
        self.rejected = True
        self.event.set()

    def reset(self):
        self.rejected = False
        self.event.set()

    async def wait(self):
        await self.event.wait()
        return not self.rejected
    


