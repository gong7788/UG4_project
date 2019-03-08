
from correctingagent.agents.agents import CorrectingAgent, Tracker, read_sentence
from correctingagent.pddl.goal_updates import create_goal_options
from correctingagent.pddl import goal_updates, pddl_functions
from correctingagent.models.pgmmodels import PGMModel, create_rules
from correctingagent.models.prob_model import KDEColourModel

def split_rule(rule):
    bits = rule.split('.(')
    red = bits[1].split('->')[0].strip()
    bits2 = bits[1].split(' (')
    blue, on = bits2[1].split('&')
    blue = blue.strip()
    on = on.replace('))', '').strip()
    return [red, blue, on]

def get_predicate_name(predicate):
    return predicate.split('(')[0]

def get_predicate_args(predicate):
    args = predicate.split('(')[1].replace(')', '')
    return [arg.strip() for arg in args.split(',')]

def get_predicate(predicate):
    pred = predicate.split('(')[0]
    args = predicate.split('(')[1].replace(')', '')
    args = [arg.strip() for arg in args.split(',')]
    return pred, args

def rule_to_pddl(rule):
    rule_split = split_rule(rule)
    red, o1 = get_predicate(rule_split[0])
    blue, o2 = get_predicate(rule_split[1])
    on, (x, y) = get_predicate(rule_split[2])


    if x == o1[0] and y == o2[0]:
        r1, r2 = create_goal_options([red], [blue])
        return r1
    if x == o2[0] and y == o1[0]:
        r1, r2 = create_goal_options([blue], [red])
        return r2
    else:
        raise ValueError('Should not get here')




class PGMCorrectingAgent(CorrectingAgent):
    def __init__(self, world, colour_models=None, rule_beliefs=None,
                 domain_file='blocks-domain.pddl', teacher=None, threshold=0.7,
                 update_negative=True, update_once=True, colour_model_type='default',
                 model_config={}, tracker=Tracker()):



        super(PGMCorrectingAgent, self).__init__(world, colour_models, rule_beliefs,
                 domain_file, teacher, threshold,
                 update_negative, update_once, colour_model_type,
                 model_config, tracker)

        self.pgm_model = PGMModel()
        self.time = 0
        self.last_correction = -1
        self.marks = set()



    def update_goal(self):
        rule_probs = self.pgm_model.get_rule_probs(update_prior=True)
        rules = []
        for rule, p in rule_probs.items():
            #print(rule, p)

            if p > 0.5:
                rules.append(rule_to_pddl(rule))
                #print(rule)

        self.goal = goal_updates.goal_from_list(rules)
        #print(self.goal.asPDDL())



    def no_correction(self, action, args):
        self.time += 1
        #print(self.time)
        if args[0] in self.marks or args[1] in self.marks:
            print('we did the thing')
            self.pgm_model.add_no_correction(args, self.time)
            data = self.get_colour_data(args)
            corr = 'corr_{}'.format(self.time)
            data[corr] = 0
            self.pgm_model.observe(data)
            self.pgm_model.infer()
            self.update_cms()



    def get_correction(self, user_input, actions, args, test=False):
        self.time += 1
        self.last_correction = self.time
        #print(self.time)

        not_on_xy = pddl_functions.create_formula('on', args, op='not')
        self.tmp_goal = goal_updates.update_goal(self.tmp_goal, not_on_xy)
        #print(self.time)
        args_for_model = args.copy()
        #print(actions, args, user_input)
        #print(read_sentence(user_input, use_dmrs=False))
        message = read_sentence(user_input, use_dmrs=False)
        args_for_model = args.copy()
        if message.T == 'table':
            args_for_model += [message.o3]
        data = self.get_colour_data(args_for_model)
        violations = self.build_pgm_model(message, args)
        corr = 'corr_{}'.format(self.time)

        data[corr] = 1
        for a in args:
            self.marks.add(a)

        red = '{}({})'.format(message.o1[0], args[0])
        blue = '{}({})'.format(message.o2[0], args[1])
        if message.T == 'table':
            redo3 = '{}({})'.format(message.o1[0], message.o3)
            blueo3 = '{}({})'.format(message.o2[0], message.o3)
            self.marks.add(message.o3)

        if 't' in args[1]:
            data[blue] = 0



        self.pgm_model.observe(data)
        q = self.pgm_model.query(list(violations))

        m_r1 = q[violations[0]]
        m_r2 = q[violations[1]]
        # q = self.pgm_model.infer(list(violations))
        #
        # m_r1 = q[violations[0]].values[1]
        # m_r2 = q[violations[1]].values[1]
        #print(m_r1, m_r2)

        print(user_input)

        if max(m_r1, m_r2) < self.threshold:
            question = 'Is the top object {}'.format(message.o1[0])
            #dialogue.info('R: ' + question)
            print(question)
            answer = self.teacher.answer_question(question, self.world)
            #dialogue.info("T: " + answer)
            print(answer)
            bin_answer = int(answer.lower() == 'yes')
            self.pgm_model.observe({red:bin_answer})
            q = self.pgm_model.query(list(violations))
            m_r1 = q[violations[0]]
            m_r2 = q[violations[1]]


        #self.update_rules()



        self.update_cms()
        self.update_goal()
        self.world.back_track()
        #super(PGMCorrectingAgent, self).get_correction(user_input, actions, args, test=test)



    def update_cms(self):
        colours = self.pgm_model.get_colour_predictions()
        for cm in self.colour_models.values():
            cm.reset()


        for colour, p in colours.items():
            colour, arg = get_predicate(colour)
            arg = arg[0]
            fx = self.get_colour_data([arg])['F({})'.format(arg)]
            #print(colour, p)
            if p > 0.7:
                self.colour_models[colour].update(fx, p)


    def add_cm(self, colour_name):
        try:
            red_cm = self.colour_models[colour_name]
        except KeyError:
            red_cm = KDEColourModel(colour_name, **self.model_config)
            self.colour_models[colour_name] = red_cm
        return red_cm

    def build_pgm_model(self, message, args):

        rules = create_rules(message.o1[0], message.o2[0])
        red_cm = self.add_cm(message.o1[0])
        blue_cm = self.add_cm(message.o2[0])

        if message.T == 'tower':
            violations = self.pgm_model.create_tower_model(rules, red_cm, blue_cm, args, self.time)
        elif message.T == 'table':
            violations = self.pgm_model.create_table_model(rules, red_cm, blue_cm, args + [message.o3], self.time)
        #print(rules)
        return violations


    def get_colour_data(self, args):
        observation = self.world.sense()
        colour_data = observation.colours
        data = {'F({})'.format(arg): colour_data[arg] for arg in args if 't' not in arg}
        return data


    def new_world(self, world):

        self.marks = set()
        self.time = 0
        self.last_correction = -1

        self.pgm_model.reset()
        for cm in self.colour_models.values():
            cm.fix()

        super(PGMCorrectingAgent, self).new_world(world)
