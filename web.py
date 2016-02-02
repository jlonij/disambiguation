import os, sys
#os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))


from bottle import abort, route, run, template, request, default_app
import disambiguation
import hashlib


@route('/link')
def link():
    os.chdir(os.path.dirname(__file__))

    #fh=open('/tmp/tmppploggg', 'a')
    #fh.write(os.getcwd()  + "\n")
    #fh.close()

    if request.params.get('ne') is not None:
        debug = request.params.get('debug')
        ne = request.params.get('ne')
        ne_type = request.params.get('ne_type')
	url = request.params.get('url')

        linker = disambiguation.Linker()
	if url and ne_type:
            link, p, mainLabel, reason = linker.link(ne, ne_type, url)
	else:
            link, p, mainLabel, reason = linker.link(ne)

	result = dict()

        if debug:
            result['ne'] = ne
            result['reason'] = reason
            fh = open('disambiguation.py', 'r')
            disambig = fh.read()
            fh.close()
            result['checksum'] = hashlib.md5(disambig).hexdigest()

        if link:
            result['ne'] = ne
            result['link'] = link[1:-1]
            result['p'] = p
            result['name'] = mainLabel

        return result
    else:
        abort(400, "No fitting argument (\"ne=...\") given.")


#run(host='localhost', port=5001)
application = default_app()
