// Remember to use static and const. Also pass by reference

// if true, a comes before b
// note when arguments are equal, comparator must return false
static bool compare(const pair<int, int> &a, const pair<int, int> &b) {
    return a.second * (log(a.first)) < b.second * (log(b.first));
}

// above comparator compares a^b and c^d