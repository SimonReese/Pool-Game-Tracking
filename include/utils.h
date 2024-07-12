#ifndef UTILS
#define UTILS

#include <string>
/**
 * TODO: remove stub functions
 */

/**
 * Testing function to greet user
 * @param username name of the user to greet
*/
void helloFunction(std::string username = "User");

/**
 * Sort tuple for key descending order
 */
bool sortTupleKeysDescending(std::tuple<double, int, bool>& first, std::tuple<double, int, bool>& second);

#endif
