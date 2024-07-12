#include "utils.h"
#include <iostream>

/**
 * TODO: remove stub functions
 */
void helloFunction(std::string username){
    // Print hello + username
    std::cout << "Hello " << username << std::endl;
}


bool sortTupleKeysDescending(std::tuple<double, int, bool> &first, std::tuple<double, int, bool> &second){
    // We want to sort tuples in decreasing order
    return std::get<0>(first) > std::get<0>(second);
}
