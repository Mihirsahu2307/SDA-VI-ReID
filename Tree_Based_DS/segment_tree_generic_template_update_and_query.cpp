const int N = 4e5 + 5; // 4 * size of array
int seg[N];
// following implementation is for 0-indexed segment tree:

void update(int in, int val, int si, int l, int r) {
    if(l == r) {
        seg[si] = val;
        return; // remember to return here
    }

    int mid = (l + r) >> 1;
    if(in <= mid) {
        update(in, val, 2 * si + 1, l, mid);
    }
    else {
        update(in, val, 2 * si + 2, mid + 1, r);
    }
    seg[si] = seg[2 * si + 1] + seg[2 * si + 2];
}

int query(int ql, int qr, int si, int l, int r) {
    if(ql > r || l > qr) {
        return 0;
    }
    if(ql <= l && r <= qr) {
        return seg[si];
    }

    int mid = (l + r) >> 1;
    return query(ql, qr, 2 * si + 1, l, mid) + query(ql, qr, 2 * si + 2, mid + 1, r);
}