// check: https://cp-algorithms.com/algebra/all-submasks.html

// To iterate over all submasks(s) of a mask(m):
// s := s - 1
// will set lsb = 0 and all bits to the right of lsb will become set


for(int s = m; ; s = (s - 1) & m) {
	if(s == 0) {
		// handle case s = 0
		break;
	}
	// use s
}