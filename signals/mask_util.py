import numpy as np
import numpy.ma as ma



def grow_mask(mask, n):
    N = len(mask)
    return [mask[max(0,i-n):min(N, i+n+1)].any() for i in range(N)]

def mask_blocks(mask):
    """
    Return a list of masked blocks (contiguous portions of the signal in which the mask is True).

    Throws an IndexError if the mask is False.
    """

    blocks = []
    block_start = 0

    try:
        in_block = mask[0]
    except:
        return []

    for i in range(len(mask)):
        if in_block and not mask[i]:        # end of a block
            blocks.append((block_start, i))
            in_block=False

        if not in_block and mask[i]:        # start of a block
            in_block=True
            block_start=i

    if in_block: # special case for blocks reaching the end of the mask
        blocks.append((block_start, i+1))

    return blocks

def mirror_missing(m):

    """
    Fills in missing values by mirroring the values to either side.
    """

    data = m.filled(m.mean())
    mask = m.mask

    try:
        blocks = mask_blocks(mask)
    except IndexError:
        return ma.masked_array(data, mask)

    for i, block in enumerate(blocks):
        start = block[0]
        end = block[1]
        n = end - start

        # we can copy forward into each block an amount of signal
        # equal to the time since the end of the previous block.
        forward_copy = start if i==0 else (start - blocks[i-1][1])

        # we can copy backwards into each block an amount of signal
        # equal to the time until the start of the next block.
        backward_copy = (len(data)) - end if i==(len(blocks)-1) else (blocks[i+1][0] - end)

        max_copy = max(forward_copy, backward_copy)
        if forward_copy >= n/2 :
            if backward_copy >= n/2:
                forward_copy = int(np.floor(n/2.0))
                backward_copy = int(np.ceil(n/2.0))
                max_copy=backward_copy
            else:
                forward_copy = min(forward_copy, n - backward_copy)
        elif backward_copy >= n/2:
            backward_copy = min(backward_copy, n - forward_copy)

        for k in range(max_copy):
            if k < forward_copy:
                data[start+k] = data[start-k-1]
            if k < backward_copy:
                data[end-k-1] = data[end+k]


    return ma.masked_array(data, mask)

