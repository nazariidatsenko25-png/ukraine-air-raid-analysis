import json
import sys
import os

log_file = sys.argv[1]
md_file = sys.argv[2]

if not os.path.exists(log_file):
    print(f"Log file not found: {log_file}")
    sys.exit(1)

with open(log_file, 'r') as f:
    lines = f.readlines()

new_content = "\n\n---\n\n## Session Continuation (UI/UX Refactor)\n\n"

for line in lines:
    try:
        data = json.loads(line)
        step_type = data.get('type')
        source = data.get('source')
        content = data.get('content')
        
        if not content:
            continue
            
        if step_type == 'USER_INPUT':
            # strip xml tags if they exist
            if "<USER_REQUEST>" in content:
                content = content.split("<USER_REQUEST>")[1].split("</USER_REQUEST>")[0].strip()
            new_content += f"### User\n\n{content}\n\n---\n\n"
        elif source == 'MODEL' and not step_type.startswith('TOOL'):
            new_content += f"### Antigravity AI\n\n{content}\n\n---\n\n"
    except Exception as e:
        pass

with open(md_file, 'a') as f:
    f.write(new_content)

print("Appended successfully.")
