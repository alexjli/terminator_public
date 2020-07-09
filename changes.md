added _VF.py, types.py to batched_attn_mask
added "from typing import Optional" etc to torch._jit_internal
changed .bool() to .byte()
added TransformerEncoder code and changed imports accordingly
converted negate_padding_mask to float before multiplying
change src_mask to float before applying, chaing src_key_padding_mask to byte before applying
