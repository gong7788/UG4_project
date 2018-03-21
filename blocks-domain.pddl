(define (domain blocksworld)
  (:requirements :strips :disjunctive-preconditions
   :negative-preconditions :quantified-preconditions
   :conditional-effects)
(:predicates (clear ?x)
             (on-table ?x)
             (arm-empty)
             (holding ?x)
             (on ?x ?y)
	           (blue ?x)
             (red ?x)
	     (in-tower ?x))


(:action put
  :parameters (?ob ?underob)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty) (clear ?underob))
  :effect (and (on ?ob ?underob) (not (clear ?underob)) (not (on-table ?ob))
               (when (in-tower ?underob) (in-tower ?ob)))
  )
)
