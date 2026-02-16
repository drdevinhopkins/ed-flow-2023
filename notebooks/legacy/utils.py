import os
import time
import datetime
import dropbox

def upload(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    # with stopwatch('upload %d bytes' % len(data)):
    try:
        res = dbx.files_upload(
            data, path, mode,
            client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
            mute=True)
        check_for_links = dbx.sharing_list_shared_links(path)
        print(check_for_links)
        # if check_for_links.links:
        #     link_to_file = check_for_links.links[0].url
        #     print('already exists', link_to_file)
        # else:
        #     shared_link_metadata = dbx.sharing_create_shared_link_with_settings(path)
        #     # print(shared_link_metadata.url)
        #     link_to_file = shared_link_metadata.url
        #     print('created new link', link_to_file)

    except dropbox.exceptions.ApiError as err:
        print('*** API error', err)
        return None
    print('uploaded as', res.name.encode('utf8'))
    return res
