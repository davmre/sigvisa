import os
import errno
import shutil

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def clear_directory(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                pass
            else:
                raise

        mkdir_p(path)

def remove_directory(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                pass
            else:
                raise


def next_unused_int_in_dir(path):
    max_int = 0
    for fname in os.listdir(path):
        try:
            max_int = max(max_int, int(fname))
        except ValueError:
            continue
    return max_int+1

def make_next_unused_dir(base_path, formatter):
    # avoid race conditions if two processes are both trying to create a new dir 
    i = next_unused_int_in_dir(base_path)
    while True:
        path = os.path.join(base_path, formatter(i))
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                i += 1
                continue
        break
    return path
