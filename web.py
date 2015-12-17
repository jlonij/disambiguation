import os, sys
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))


from bottle import abort, route, run, template, request, default_app
import disambiguation
import hashlib


@route('/link')
def link():
    if request.params.get('ne') is not None:
        debug = request.params.get('debug')
        ne = request.params.get('ne')
        ne_type = request.params.get('ne_type')
	url = request.params.get('url')
	
	if url and ne_type:
	    link, p, mainLabel, reason = disambiguation.linkEntity(ne, ne_type, url)
	else:
	    link, p, mainLabel, reason = disambiguation.linkEntity(ne)

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


