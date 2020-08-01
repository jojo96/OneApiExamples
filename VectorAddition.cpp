#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <CL/sycl/intel/fpga_extensions.hpp>

using namespace cl::sycl;
using namespace std;

constexpr access::mode sycl_read = access::mode::read;
//sycl_read is a constant expression which denotes read mode

constexpr access::mode sycl_write = access::mode::write;
//sycl_write is a constant expression which denotes write mode

class VectorAdd;


static const size_t ARRAY_SIZE = 5; // ARRAY_SIZE denotes size of array


void initialize_array(array<cl_int, ARRAY_SIZE>& arr);//initialize_array function signature


void add_vectors_parallel(std::array<cl_int, ARRAY_SIZE>& sum_array,
    const array<cl_int, ARRAY_SIZE>& addend_array_1,
    const array<cl_int, ARRAY_SIZE>& addend_array_2);
//add_vectors_parallel uses parallel for loop to add 2 vectors. A vector is a 1-D array



int main() {
    array<cl_int, ARRAY_SIZE> addend_array_1;
    array<cl_int, ARRAY_SIZE> addend_array_2;
    array<cl_int, ARRAY_SIZE> sum_array_parallel;
  

  
    initialize_array(addend_array_1);//addend_array_1= {1,2,3,4,5}
    initialize_array(addend_array_2);//addend_array_2= {1,2,3,4,5}
    initialize_array(sum_array_parallel);
  

  
    add_vectors_parallel(sum_array_parallel, addend_array_1, addend_array_2);
   
   for(size_t i=0;i<ARRAY_SIZE;i++)
   {
        cout<<sum_array_parallel[i];
        cout<<"\n";
    }
   
    cout << "success" << "\n";
    return 0;
}

void initialize_array(array<cl_int, ARRAY_SIZE>& arr) {
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = i;
    }
}

void add_vectors_parallel(array<cl_int, ARRAY_SIZE>& sum_array,
    const array<cl_int, ARRAY_SIZE>& addend_array_1,
    const array<cl_int, ARRAY_SIZE>& addend_array_2) {

  //this part is for device selection. There is also an automatic option
#ifdef INTEL_FPGA
    // FPGA device selector:  Emulator or Hardware
#ifdef FPGA_EMULATOR
    intel::fpga_emulator_selector device_selector;#else
    intel::fpga_selector device_selector;
#endif
#else
       
    default_selector device_selector;

#endif

    queue device_queue(device_selector);

    cout << "Device: "
        << device_queue.get_device().get_info<info::device::name>()
        << "\n";
//printing selected device name with cout
   
    range<1> num_items{ ARRAY_SIZE };//range<1> to create 1-D array of size ARRAY_SIZE

   //below: creating 3 buffers for 3 arrays
    buffer<cl_int, 1> addend_1_buf(addend_array_1.data(), num_items);
    buffer<cl_int, 1> addend_2_buf(addend_array_2.data(), num_items);
    buffer<cl_int, 1> sum_buf(sum_array.data(), num_items);//sum_buf has the sum of 2 vectors

  
    device_queue.submit([&](handler& cgh) {
       
        auto addend_1_accessor = addend_1_buf.template get_access<sycl_read>(cgh);
        auto addend_2_accessor = addend_2_buf.template get_access<sycl_read>(cgh);

     
        auto sum_accessor = sum_buf.template get_access<sycl_write>(cgh);

      
        cgh.parallel_for<class VectorAdd>(num_items, [=](id<1> wiID) {
            sum_accessor[wiID] = addend_1_accessor[wiID] + addend_2_accessor[wiID];
        });//parallel for loop where parallel processing occurs
    });
 
}
