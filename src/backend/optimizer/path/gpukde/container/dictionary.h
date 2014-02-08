#ifndef DICTIONARY_H
#define DICTIONARY_H

typedef struct dictionary_entry {
   char* key;
   void* payload;
   struct dictionary_entry* next;
} dictionary_entry_t;

typedef struct dictionary {
   dictionary_entry_t** hash_table;
   unsigned int hash_table_size;
} dictionary_tt;

typedef dictionary_tt* dictionary_t;

/**
 * Initializes a new dictionary data structure.
 * @return The newly initialized dictionary.
 */
dictionary_t dictionary_init(void);

/**
 * Releases a dictionary.
 * @param dictionary The dictionary that should be released.
 * @param release_payloads Flag indicating whether dictionary payloads should also be released.
 */
void dictionary_release(dictionary_t dictionary, char release_payloads);

/**
 * Returns the payload that is associated with a given key (or NULL if the key is
 * not stored in the dictionary).
 * @param dictionary
 * @param key
 * @return The registered payload for this key, or NULL if the key is unknown.
 */
void* dictionary_get(dictionary_t dictionary, const char* key);

/**
 * Inserts a new entry into the dictionary. If the key is already known, the existing
 * payload is being overwritten.
 * @param dictionary
 * @param key
 * @param payload
 * @return The previous payload that was registered for the key, or NULL if the key
 *         was unknown so far.
 */
void* dictionary_insert(dictionary_t dictionary, const char* key, void* payload);

/**
 * Removes a key from the dictionary.
 * @param dictionary
 * @param key
 * @param release_payload Flag indicating whether the dictionary payload should also be released.
 */
void dictionary_remove(dictionary_t dictionary, const char* key, char release_payload);

// Iterator for the dictionary. Usage is as follows:
// dictionary_iterator_t it = dictionary_iterator_init(dictionary);
// while (dictionary_iterator_key(dictionary, it)) {
//   printf("%s -> %p\n", dictionary_iterator_key(dictionary, it),
//                        dictionary_iterator_value(dictionary, it));
//   it = dictionary_iterator_next(dictionary, it);
// }
typedef struct dictionary_iterator {
  unsigned int current_bucket;
  dictionary_entry_t* current_list_element;
} dictionary_iterator_t;

/**
 * Initializes an iterator for a dictionary. After this call the iterator
 * points to the first entry in the dictionary (or nowhere, if the
 * dictionary is empty).
 *
 * @param dictionary
 * @returns An initialized dictionary_iterator_t structure.
 */
dictionary_iterator_t dictionary_iterator_init(dictionary_t dictionary);

/**
 * Advances the iterator by one position.
 *
 * @param dictionary
 * @param iterator
 * @returns The iterator pointing to the next occupied dictionary position.
 */
dictionary_iterator_t dictionary_iterator_next(dictionary_t dictionary,
                                               dictionary_iterator_t iterator);

/**
 * Returns the key at the current iterator position within the dictionary (or
 * NULL, if the iterator is empty).
 *
 * @param dictionary
 * @param iterator
 * @returns The key for the position currently pointed at by the iterator, or
 *          NULL if the iterator points nowhere.
 */
const char* dictionary_iterator_key(dictionary_t dictionary,
                                    dictionary_iterator_t iterator);

/**
 * Returns the value at the current iterator position within the dictionary (or
 * NULL, if the iterator is empty).
 *
 * @param dictionary
 * @param iterator
 * @returns The value for the position currently pointed at by the iterator, or
 *          NULL if the iterator points nowhere.
 */
void* dictionary_iterator_value(dictionary_t dictionary,
                                dictionary_iterator_t iterator);

#endif /* DICTIONARY_H */
