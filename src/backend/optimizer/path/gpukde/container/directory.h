/* 
 * File:   directory.h
 * Author: mheimel
 * 
 * A self-growing key-value directory data structure that stores
 * arbitrary payloads using a sorted list of fixed-size keys.
 * 
 * Before using the directory, it needs to be allocated using:
 *    directory_t directory_init(size_t key_size, unsigned int initial_capacity)
 * directory_init needs the size in bytes of the keys that will be stored in 
 * the directory. Once the directory is no longer needed, it must be discarded 
 * using:
 *    void directory_release(directory_t set, char release_payload)
 * If release_payload is set, the cleanup will also release all payloads.
 * 
 * There are three standard functions to manipulate the set:
 *    void* directory_insert(directory_t directory, void* key, void* payload);
 *       Inserts a kvp into the directory. If another payload was already
 *       registered for the key, the old payload will be overwritten and 
 *       returned.
 *    void directory_remove(set_t set, void* key, char release_payload) 
 *       Removes a kvp from the set. If release_payload is set, this will
 *       also remove all references
 * 
 * Example Usage:
 *    unsigned int k1 = 1; unsigned int v1 = 10;
 *    unsigned int k2 = 2; unsigned int v2 = 20;
 *    
 *    directory_t directory = directory_init (sizeof(unsigned int), 10);
 *    directory_insert(directory, &k1, &v1); [1,1]
 *    directory_insert(directory, &k1, &v2); [1,2]
 *    directory_insert(directory, &k2, &v1); [1,2],[2,1]
 *    directory_remove(directory, &k1, 0);   [2,1]
 *    directory_release(directory, 0),
 *
 * Created on 7. Juni 2013, 19:00
 */

#ifndef DIRECTORY_H
#define DIRECTORY_H

#include <stdlib.h>

// Utility class that implements a sorted list of key value pairs.
typedef struct {
    // Size of a single key in bytes.
    size_t key_size;
    // Capacity of the directory in nr of keys.
    unsigned int capacity;
    // How many entries are stored in the directory.
    unsigned int entries;
    // Sorted byte array of keys in the directory.
    unsigned char* keys;
    // Array of payloads (has identical order to key array).
    void** payloads;
    // Position of the last access.
    unsigned int last_access;
} directory_tt;

typedef directory_tt* directory_t;

// Initialize a directory for the given key size.
directory_t directory_init(size_t key_size, unsigned int initial_capacity);
// Clear a directory. If release_payloads is true, the function will also
// delete all registered payloads.
void directory_release(directory_t directory, char release_payloads);

// Clears all entries in the directory.
void directory_clear(directory_t directory, char release_payloads);

// Accessor functions to get the key / value at a given offset position.
void* directory_keyAt(directory_t directory, unsigned int position);
void* directory_valueAt(directory_t directory, unsigned int position);

// Find a given key in the directory. Returns directory->entries if the key
// was not found.
unsigned int directory_find(directory_t directory, const void* key);
// Return the payload for a given key.
void* directory_fetch(directory_t directory, const void* key);
// Macro to fetch the payload for a given key as a given type.
#define DIRECTORY_FETCH(directory, key, type) ((type*)(directory_fetch((directory), (key))))

// Insert a key into the directory.
// Returns the old payload at this position, or NULL if no other payload was stored there.
void* directory_insert(directory_t directory, const void* key, void* payload);
// Deletes a key from the directory. If release_payload is true, the function 
// will also delete the registre dpayload for the key.
void directory_remove(directory_t directory, const void* key, char release_payload);

#endif	/* DIRECTORY_H */
