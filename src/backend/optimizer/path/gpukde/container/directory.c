// Inhibit gcc warnings on mixed declarations
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

#include <string.h>

#include "directory.h"

// Initialize a directory for the given key size.
directory_t directory_init(size_t key_size, unsigned int initial_capacity) {
    directory_tt* result = malloc(sizeof(directory_tt));
    result->capacity = initial_capacity;
    result->key_size = key_size;
    result->entries = 0;
    result->payloads = malloc(sizeof(void*)*result->capacity);
    result->keys = malloc(key_size*result->capacity);
    result->last_access = 0;    
    return result;
}

// Release a given directory. 
void directory_release(directory_t directory, char release_payloads) {
    if (release_payloads) {
        unsigned int i;
        for (i=0; i<directory->entries; ++i) {
            free(directory->payloads[i]);
        }
    }
    free(directory->keys);
    free(directory->payloads);
    free(directory);
}

void directory_clear(directory_t directory, char release_payloads) {
    if (release_payloads) {
        unsigned int i;
        for (i=0; i<directory->entries; ++i) {
            free(directory->payloads[i]);
            directory->payloads[i] = NULL;
        }
    }
    directory->entries = 0;
}

// Helper function that compares a key against the directory key at a given position.
// The function returns:
//    -1 if key < directory[pos]
//     0 if key = directory[pos]
//     1 if key > directory[pos]
static int dir_keycmp(directory_t directory, const unsigned char* key,
                      unsigned int pos) {
    int i;
    int cmp = 0;
    for (i=directory->key_size - 1; cmp == 0 && i>=0; --i) {
        if (key[i] < directory->keys[pos*directory->key_size + i])
            cmp = -1;
        else if (key[i] > directory->keys[pos*directory->key_size + i])
            cmp = 1;
    }
    return cmp;
}

// Helper function that performs a binary search on the directory.
// The function will return the first position p, such that A[p] >= key.
static unsigned int dir_bsearch(directory_t directory, const unsigned char* key) {
    if (directory->entries == 0)
        return 0; // Handle the empty case.
    unsigned int l = 0;
    unsigned int h = directory->entries - 1;
    // Binary search:
    while (h >= l) {
      // Compute the mid-point.
      unsigned int mid = (l + h) / 2; // Would overflow for very large l,h. However, we expect the directories to remain reasonably small.
      // Compare the key with the mid-point.
      int cmp = dir_keycmp(directory, key, mid);
      if (cmp < 0) {
          if (mid == 0)
              return 0; // Prevent an underflow (happens for 1-element arrays if we search for a key that is smaller than the only element)
          h = mid - 1;
      } else if (cmp > 0) {
          l = mid + 1;
      } else {
          // We found the position.
          return mid;
      }
    }
    return l;  // Lower bound for insert.
}

// Locate an entry in the directory. 
unsigned int directory_find(directory_t directory, const void* key) {
    const unsigned char* ckey = (const unsigned char*)key;
    if (directory->entries == 0)
        return directory->entries;  // Handle empty case.
    // Check if the last accessed position corresponds to this key.
    if (directory->last_access < directory->entries 
            && dir_keycmp(directory, ckey, directory->last_access) == 0)
        return directory->last_access;
    // If it was not, use binary search to find it.
    unsigned int pos = dir_bsearch(directory, ckey);
    // Check if this is actually the key position, as bsearch might return
    // an upper bound for the key position as well.
    if (dir_keycmp(directory, ckey, pos) == 0) {
        directory->last_access = pos;
        return pos;
    }
    // Indicate KEY_NOT_FOUND.
    return directory->entries;
}

void* directory_keyAt(directory_t directory, unsigned int position) {
    return &(directory->keys[directory->key_size * position]);
}

void* directory_valueAt(directory_t directory, unsigned int position) {
    return directory->payloads[position];
}

void* directory_fetch(directory_t directory, const void* key) {
    unsigned int pos = directory_find(directory, key);
    if (pos==directory->entries)
        return NULL;
    else
        return directory->payloads[pos];
}

// Insert a payload at a given position.
static void* directory_put(directory_t directory, unsigned int position, void* payload) {
    if (position < directory->entries) {
        void* tmp = directory->payloads[position];
        directory->payloads[position] = payload;
        return tmp;
    }
    return NULL;
}

// Insert a new entry in the directory.
void* directory_insert(directory_t directory, const void* key, void* payload) {
    const unsigned char* ckey = (const unsigned char*)key;
    // Use bsearch to find the position of the key.
    unsigned int insert_position = dir_bsearch(directory, ckey);
    // If the key is already stored in the directory, replace it.
    if (insert_position < directory->entries 
            && dir_keycmp(directory, ckey, insert_position) == 0) {
        return directory_put(directory, insert_position, payload);
    } 
    // If it is not, we have to insert the key, make sure that there is enough
    // storage left in the directory.
    if (directory->capacity <= directory->entries) {
        // We dont have any capacity left in the directory, grow it first.
        directory->capacity += 10;
        directory->keys = realloc(directory->keys, 
                directory->key_size*directory->capacity);
        directory->payloads = 
                realloc(directory->payloads, sizeof(void*)*directory->capacity);
    }
    // Now move the keys and payloads forward to make place for the new
    // insertion.
    memmove(&(directory->keys[(insert_position + 1)*directory->key_size]), 
            &(directory->keys[insert_position * directory->key_size]),
            (directory->entries - insert_position)*directory->key_size);
    memmove(&(directory->payloads[insert_position+1]), 
            &(directory->payloads[insert_position]), 
            (directory->entries - insert_position)*sizeof(void*));
    // Ok, insert the value.
    memcpy(&(directory->keys[insert_position * directory->key_size]), ckey,
            directory->key_size);
    directory->payloads[insert_position] = payload;
    directory->entries++;
    return NULL;
}

void directory_remove(directory_t directory, const void* key, char release_payload) {
    const unsigned char* ckey = (const unsigned char*)key;
    // Find the position of the key.
    unsigned int position = dir_bsearch(directory, ckey);
    if (position >= directory->entries)
        return;   // Protect against empty deletion.
    if (dir_keycmp(directory, ckey, position) != 0)
        return;   // The key is not contained.
    if (release_payload)
        free(directory->payloads[position]);
    // Move all remaining entries one position to the left.
    memmove(&(directory->keys[position*directory->key_size]), 
            &(directory->keys[(position + 1) * directory->key_size]),
            (directory->entries - position - 1)*directory->key_size);
    memmove(&(directory->payloads[position]), 
            &(directory->payloads[position + 1]), 
            (directory->entries - position - 1)*sizeof(void*));
    directory->entries--;
}
