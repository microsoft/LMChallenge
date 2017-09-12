from .. import errors
import time


if __name__ == '__main__':
    with open('example/words.txt') as f:
        words = set(line.rstrip('\r\n') for line in f)
    print('Number of words: %d' % len(words))

    t0 = time.time()
    search = errors.Search(words, 20)
    t1 = time.time()
    print('Build: %.3f s' % (t1 - t0))

    # ascending word lengths
    total = 0
    for word in ['I', 'am', 'the', 'bird', 'which', 'should', 'already',
                 'increase', 'published', 'investment']:
        t0 = time.time()
        print('\t%s -> %s' % (word, ' '.join(search(word))))
        t1 = time.time()
        total += t1 - t0
        print('Search: %.3f s' % (t1 - t0))
    print('Total: %.3f s' % total)
