class ArithmeticEncoder:
    def __init__(self, bout: BitOutputStream):
        self.bout = bout
        self.low = 0
        self.high = (1 << 32) - 1
        self.pending_bits = 0   

    def update(self, cum, sym):
    # cum: cumulative freq array of length (n_tokens+1), total = cum[-1]
        total = cum[-1]
        range_ = self.high - self.low + 1
        sym_low = cum[sym]
        sym_high = cum[sym+1]
        self.high = self.low + (range_ * sym_high // total) - 1
        self.low = self.low + (range_ * sym_low // total)


        while True:
            if (self.high & (1 << 31)) == (self.low & (1 << 31)):
                bit = (self.high >> 31) & 1
                self.bout.write_bit(bit)
            # flush pending
                for _ in range(self.pending_bits):
                    self.bout.write_bit(1 - bit)
                    self.pending_bits = 0
                    self.low = ((self.low << 1) & ((1 << 32) - 1))
                    self.high = ((self.high << 1) & ((1 << 32) - 1)) | 1
            elif (self.low & (1 << 30)) and not (self.high & (1 << 30)):
                # underflow
                self.pending_bits += 1
                self.low = (self.low << 1) & ((1 << 32) - 1)
                self.high = ((self.high << 1) & ((1 << 32) - 1)) | 1
                self.low ^= (1 << 31)
                self.high ^= (1 << 31)
            else:
                break


    def finish(self):
    # emit one more bit plus pending complement bits
        bit = (self.low >> 31) & 1
        self.bout.write_bit(bit)
        for _ in range(self.pending_bits):
            self.bout.write_bit(1 - bit)
        self.bout.flush()