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
	         (in-tower ?x ?t)
             (done ?x)
             (green ?x)
             (yellow ?x)
             (pink ?x)
             (darkred ?x)
             (maroon ?x)
             (firebrick ?x)
             (crimson ?x)
             (olivedrab ?x)
             (yellowgreen ?x)
             (darkolivegreen ?x)
             (greenyellow ?x)
             (charteuse ?x)
             (lawngreen ?x)
             (palegreen ?x)
             (forestgreen ?x)
             (limegreen ?x)
             (seagreen ?x)
             (cornflowerblue ?x)
             (royalblue ?x)
             (midnightblue ?x)
             (navy ?x)
             (darkblue ?x)
             (mediumblue ?x)
             (dodgerblue ?x)
             (deepskyblue ?x)
             (lightyellow ?x)
             (lightgoldenrodyellow ?x)
             (indigo ?x)
             (darkorchid ?x)
             (darkviolet ?x)
             (rebeccapurple ?x)
             (purple ?x)
             (orange ?x)
             (block ?x)
             (tower ?x))


    (:functions (blue-count ?tower)
                (red-count ?tower)
                (green-count ?tower)
                (orange-count ?tower)
                (purple-count ?tower)
                (pink-count ?tower)
                (yellow-count ?tower))

(:action put
  :parameters (?ob ?underob ?tower)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty) (clear ?underob) (in-tower ?underob ?tower))
  :effect (and
  (on ?ob ?underob)
  (not (clear ?underob))
  (not (on-table ?ob))
  (in-tower ?ob ?tower)
  (done ?ob)
  (when (blue ?ob) (increase (blue-count ?tower) 1))
  (when (red ?ob) (increase (red-count ?tower) 1))
  (when (green ?ob) (increase (green-count ?tower) 1))
  (when (orange ?ob) (increase (orange-count ?tower) 1))
  (when (purple ?ob) (increase (purple-count ?tower) 1))
  (when (pink ?ob) (increase (pink-count ?tower) 1))
  (when (yellow ?ob) (increase (yellow-count ?tower) 1))
  )
 )
)
