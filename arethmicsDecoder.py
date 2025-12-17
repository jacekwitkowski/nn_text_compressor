import bitInputStream
from bitInputStream import BitInputStream

class ArithmeticDecoder:
    def __init__(self, binp: BitInputStream):
        self.binp = binp
        self.low = 0
        self.high = (1 << 32) - 1
        self.code = 0
        for _ in range(32):
            b = self.binp.read_bit()
            self.code = (self.code << 1) | (0 if b is None else b)

    def get_symbol(self, cum):
        total = cum[-1]
        range_ = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // range_
        # find symbol s.t. cum[s] <= value < cum[s+1]
        # linear search (256 values) is OK here
        s = 0
        while cum[s+1] <= value:
            s += 1
        # update
        sym_low = cum[s]
        sym_high = cum[s+1]
        self.high = self.low + (range_ * sym_high // total) - 1
        self.low = self.low + (range_ * sym_low // total)

        while True:
            if (self.high & (1 << 31)) == (self.low & (1 << 31)):
                self.low = ((self.low << 1) & ((1 << 32) - 1))
                self.high = ((self.high << 1) & ((1 << 32) - 1)) | 1
                b = self.binp.read_bit()
                self.code = ((self.code << 1) & ((1 << 32) - 1)) | (0 if b is None else b)
            elif (self.low & (1 << 30)) and not (self.high & (1 << 30)):
                self.low = (self.low << 1) & ((1 << 32) - 1)
                self.high = ((self.high << 1) & ((1 << 32) - 1)) | 1
                self.code = ((self.code << 1) & ((1 << 32) - 1)) | (0 if self.binp.read_bit() is None else self.binp.read_bit())
                self.low ^= (1 << 31)
                self.high ^= (1 << 31)
                self.code ^= (1 << 31)
            else:
                break
        return s