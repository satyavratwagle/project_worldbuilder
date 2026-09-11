import utils.prompts as prompts

args_dict = dict()
args_dict['user_query'] = '(My Query Here)'
args_dict['local_context'] = '(My Context Here)'
args_dict['existing_summary'] = ''

hist = prompts.get_reasoned_generation_prompt(args_dict)

for d in hist:
	print(d['content'])

'''
"prompt_name":
		{
			"system":
				{

				},
			"user":
				{

				}			
		}

'''