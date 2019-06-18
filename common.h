#define sect       	4
#define N		28
const int sm = N/sect, CORES = sect*sect;
const float ee =        2.71828;

char const img[] = "/home/parallella/my_codes/cnn/src/zero";

char const conv1_b[] = "/home/parallella/my_codes/cnn/params/c1b";
char const conv1_w[] = "/home/parallella/my_codes/cnn/params/c1w";
char const conv2_b[] = "/home/parallella/my_codes/cnn/params/c2b";
char const conv2_w[] = "/home/parallella/my_codes/cnn/params/c2w";
char const conv3_b[] = "/home/parallella/my_codes/cnn/params/c3b";
char const conv3_w[] = "/home/parallella/my_codes/cnn/params/c3w";

char const dense1_b[] = "/home/parallella/my_codes/cnn/params/d1b";
char const dense1_w[] = "/home/parallella/my_codes/cnn/params/d1w";
char const dense2_b[] = "/home/parallella/my_codes/cnn/params/d2b";
char const dense2_w[] = "/home/parallella/my_codes/cnn/params/d2w";

#define c1w_len		32*1*5*5
#define c2w_len		32*32*5*5
#define c3w_len		64*32*3*3
#define c4w_len		64*64*3*3
#define c1b_len		32
#define c2b_len		32
#define c3b_len		64
#define c4b_len		64
