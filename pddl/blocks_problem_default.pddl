(define (problem blocks-1)
  (:domain blocksworld)
  (:objects
    b1
    b2
    b3
    b4
    b5)

  (:init

   ;; We have to specify the adjacency relation on the set of
   ;; squares completely (though see slidetile.pddl for an
   ;; example of how positions in a grid can be represented
   ;; more compactly).

   (on-table b1)
   (on-table b2)
   (on-table b3)
   (on-table b4)
   (on-table b5)
   (clear b1)
   (clear b2)
   (clear b3)
   (clear b4)
   (clear b5)
   (arm-empty)
   (blue b1)
   (red b3)
   (in-tower b4)
   (green b2)
   (yellow b4)
   (red b5)
)

  ;; The goal is for the agent to get the gold and make it safely
  ;; back to the starting square.
  (:goal
     (forall (?x) (in-tower ?x))
     ;(on b1 b4)
  )
)
