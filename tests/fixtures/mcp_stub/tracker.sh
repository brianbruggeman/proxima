#!/bin/sh
inits=0
ready=0
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      inits=$((inits + 1))
      printf '%s\n' '{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"tracker","version":"1"}}}'
      ;;
    *'"method":"notifications/initialized"'*)
      ready=1
      ;;
    *'"method":"tools/call"'*)
      if [ "$ready" = 1 ]; then
        rest=${line#*'"arguments":{"id":"'}
        id=${rest%%'"'*}
        printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"inits='"$inits"'"}],"structuredContent":{"ticket":"'"$id"'","repo":"brianbruggeman/proxima","pr":42}}}'
      else
        printf '%s\n' '{"jsonrpc":"2.0","id":1,"error":{"code":-32002,"message":"server not initialized"}}'
      fi
      ;;
    *'"method":"exit"'*)
      exit 0
      ;;
  esac
done
